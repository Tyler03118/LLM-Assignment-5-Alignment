import os
import json
import math
import random
import argparse
import torch
import wandb
from tqdm import tqdm
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# 导入核心组件
from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ================= 1. vLLM 猴子补丁与初始化 (讲义提供) =================

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """在指定的 GPU 上启动 vLLM 实例"""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy, llm: LLM):
    """将训练好的策略模型权重同步给 vLLM"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# ================= 2. 辅助函数 =================

def format_prompt(question):
    """使用 R1-Zero 提示词模板"""
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
        f"User: {question}\nAssistant: <think>"
    )

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]

def run_hf_evaluation(model, tokenizer, val_file_path, reward_fn, val_limit=200):
    """原生 HF 评估 (与 SFT 相同)"""
    val_data = load_dataset(val_file_path)[:val_limit]
    model.eval()
    total_reward = 0.0
    
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    batch_size = 4
    for i in tqdm(range(0, len(val_data), batch_size), desc="评估中"):
        batch = val_data[i:i+batch_size]
        prompts = [format_prompt(d.get("prompt") or d.get("problem")) for d in batch]
        gts = [d.get("ground_truth") or d.get("expected_answer") for d in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        prompt_len = inputs["input_ids"].shape[1]
        for j, output_ids in enumerate(outputs):
            gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            if "</answer>" not in gen_text: gen_text += "</answer>"
            result = reward_fn(gen_text, gts[j])
            total_reward += result.get("answer_reward", 0.0)
            
    tokenizer.padding_side = orig_pad_side
    return (total_reward / len(val_data)) * 100

def evaluate_vllm(llm: LLM, val_file_path, reward_fn, val_limit=200):
    """使用 vLLM 进行极速评估 """
    val_data = load_dataset(val_file_path)[:val_limit]
    prompts = [format_prompt(d.get("prompt") or d.get("problem")) for d in val_data]
    gts = [d.get("ground_truth") or d.get("expected_answer") for d in val_data]
    
    # 讲义要求的评估超参数 [cite: 141-144]
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    print(f"🚀 正在使用 vLLM 极速评估 {len(prompts)} 条数据...")
    outputs = llm.generate(prompts, eval_sampling_params)
    
    total_reward = 0.0
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        if "</answer>" not in gen_text: 
            gen_text += "</answer>"
        result = reward_fn(gen_text, gts[i])
        total_reward += result.get("answer_reward", 0.0)
        
    return (total_reward / len(val_data)) * 100

# ================= 3. 主循环 (Expert Iteration) =================

def train_ei(args):
    run_name = f"ei-Db{args.db_size}-G{args.G}-ep{args.sft_epochs}"
    wandb.init(project="cs336-alignment-rl", name=run_name, config=vars(args))
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="ei_step")
    wandb.define_metric("eval/*", step_metric="ei_step")

    print("💿 加载 Tokenizer 和 策略模型到 cuda:0 ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda:0")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    print("🚀 初始化 vLLM 实例到 cuda:1 ...")
    # 注意这里指定了 cuda:1 [cite: 388]
    llm = init_vllm(args.model_id, device="cuda:1", seed=42, gpu_memory_utilization=0.85)
    
    train_data = load_dataset(args.train_data)
    
    print("\n📊 初始 Baseline 评估 (Step 0)...")
    load_policy_into_vllm_instance(policy_model, llm) 
    base_acc = evaluate_vllm(llm, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
    wandb.log({"eval/accuracy": base_acc, "ei_step": 0})
    print(f"🎯 Baseline 准确率: {base_acc:.2f}%")

    for step in range(1, args.n_ei_steps + 1):
        print(f"\n{'='*20} 专家迭代 Step {step}/{args.n_ei_steps} {'='*20}")
        
        # 1. 采样 Db 个问题 [cite: 432]
        batch_data = random.sample(train_data, args.db_size)
        prompts = [format_prompt(d.get("problem") or d.get("prompt")) for d in batch_data]
        gts = [d.get("expected_answer") or d.get("ground_truth") for d in batch_data]
        
        # 2. 同步策略模型权重到 vLLM [cite: 392]
        print("🔄 正在同步权重到 vLLM...")
        load_policy_into_vllm_instance(policy_model, llm)
        
        # 3. vLLM 批量采样 G 个 Rollouts [cite: 434, 440-448]
        print(f"🎲 正在为 {args.db_size} 个问题各自生成 {args.G} 个解答 (共 {args.db_size * args.G} 个)...")
        sampling_params = SamplingParams(
            n=args.G,
            temperature=1.0,
            max_tokens=1024,
            min_tokens=4,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            seed=42 + step
        )
        vllm_outputs = llm.generate(prompts, sampling_params)
        
        # 4. 奖励验证与过滤 [cite: 435]
        sft_prompts, sft_responses = [], []
        correct_count = 0
        
        print("🔍 正在过滤正确的专家轨迹 (r=1)...")
        for i, output in enumerate(vllm_outputs):
            gt = gts[i]
            for req_output in output.outputs:
                text = req_output.text
                if "</answer>" not in text: text += "</answer>"
                
                # Rule-based 评分
                reward = r1_zero_reward_fn(text, gt)
                if reward.get("answer_reward", 0.0) == 1.0:
                    sft_prompts.append(prompts[i])
                    sft_responses.append(text)
                    correct_count += 1
                    
        pass_rate = (correct_count / (args.db_size * args.G)) * 100
        print(f"✅ 过滤完成！生成了 {correct_count} 条正确轨迹 (通过率: {pass_rate:.2f}%)。")
        wandb.log({"train/rollout_pass_rate": pass_rate, "train/sft_dataset_size": correct_count, "ei_step": step})
        
        if correct_count == 0:
            print("⚠️ 本轮没有生成任何正确的轨迹，跳过 SFT 训练！")
            continue
            
        # 5. 使用过滤后的数据进行 SFT 训练 [cite: 435]
        print(f"🔥 开始本轮 SFT 训练 (Epochs: {args.sft_epochs})...")
        policy_model.train()
        
        epoch_losses, epoch_entropies = [], []
        for epoch in range(args.sft_epochs):
            # 打乱数据
            combined = list(zip(sft_prompts, sft_responses))
            random.shuffle(combined)
            sft_prompts_shuffled, sft_responses_shuffled = zip(*combined) if combined else ([], [])
            
            for i in tqdm(range(0, len(sft_prompts_shuffled), args.train_batch_size), desc=f"SFT Epoch {epoch+1}"):
                batch_p = sft_prompts_shuffled[i : i + args.train_batch_size]
                batch_r = sft_responses_shuffled[i : i + args.train_batch_size]
                if len(batch_p) < args.micro_batch_size: continue
                    
                optimizer.zero_grad()
                current_accum_steps = math.ceil(len(batch_p) / args.micro_batch_size)
                
                step_loss = 0
                step_entropy = 0
                
                for j in range(0, len(batch_p), args.micro_batch_size):
                    mb_p = batch_p[j : j + args.micro_batch_size]
                    mb_r = batch_r[j : j + args.micro_batch_size]
                    
                    data = tokenize_prompt_and_output(mb_p, mb_r, tokenizer)
                    input_ids = data["input_ids"].to("cuda:0")
                    labels = data["labels"].to("cuda:0")
                    mask = data["response_mask"].to("cuda:0")
                    
                    # 获取 Log Probs 并要求返回 Entropy [cite: 297]
                    outputs = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=True)
                    loss, meta = sft_microbatch_train_step(outputs["log_probs"], mask, current_accum_steps)
                    
                    # 计算 Token 平均熵 [cite: 263-276]
                    if "token_entropy" in outputs:
                        token_entropy = outputs["token_entropy"]
                        valid_entropy = (token_entropy * mask).sum() / mask.sum().clamp(min=1e-5)
                        step_entropy += valid_entropy.item() / current_accum_steps
                    
                    step_loss += loss.item()
                
                # 梯度裁剪 [cite: 449]
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(step_loss)
                if step_entropy > 0: epoch_entropies.append(step_entropy)
                
        # 记录 SFT 阶段的平均 Loss 和 Entropy [cite: 454]
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_entropy = sum(epoch_entropies) / len(epoch_entropies) if epoch_entropies else 0
        wandb.log({"train/sft_loss": avg_loss, "train/response_entropy": avg_entropy, "ei_step": step})
        
        # 6. 本轮结束，评估并保存
        print("🔄 评估前同步最新权重到 vLLM...")
        load_policy_into_vllm_instance(policy_model, llm)
        
        # 使用 vLLM 极速评估
        acc = evaluate_vllm(llm, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
        wandb.log({"eval/accuracy": acc, "ei_step": step})
        print(f"🎯 Step {step} 准确率: {acc:.2f}%")
        
        ckpt_path = os.path.join(args.output_dir, run_name, f"step-{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        policy_model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        
    print("✅ 专家迭代 (EI) 训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Expert Iteration (EI)")
    # 强烈建议将 model_id 设置为你刚训练好拿到 19% 准确率的 checkpoint 路径
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B") 
    parser.add_argument("--train_data", type=str, default="data/MATH/train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/ei_runs")
    
    # EI 核心超参数 
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--db_size", type=int, default=512, help="每次迭代采样的问题数量")
    parser.add_argument("--G", type=int, default=8, help="每个问题生成的 Rollout 数量")
    parser.add_argument("--sft_epochs", type=int, default=2, help="在正确轨迹上训练的 Epoch 数")
    
    # 训练超参数
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--val_limit", type=int, default=200)
    
    args = parser.parse_args()
    train_ei(args)