import os
import json
import math
import argparse
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import SamplingParams, LLM
from unittest.mock import patch

def vllm_set_random_seed(seed):
    """万能随机种子设置，不依赖任何特定版本的 vLLM 内部 API"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# 导入你写的核心组件
from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ================= 0. vLLM 兼容性工具函数 (v0.5.4 稳定版) =================

def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.2):
    """最原始、最干净的 vLLM 初始化 (适用于 v0.5.4)"""
    vllm_set_random_seed(seed)
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """完美适配 vLLM 0.5.4 的权重同步逻辑"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ================= 1. 辅助函数：数据加载与评估 =================

def load_my_sft_data(file_path, limit=None):
    """加载 JSONL 数据，支持限制数量"""
    prompts = []
    responses = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if limit is not None:
            lines = lines[:limit]
            
        for line in lines:
            data = json.loads(line)
            prompts.append(data["prompt"])
            responses.append(data["response"])
            
    print(f"✅ 成功从 {file_path} 加载了 {len(prompts)} 条训练数据。")
    return prompts, responses


def run_vllm_evaluation(llm, val_file_path, reward_fn):
    """使用 vLLM 进行极速评估"""
    if not os.path.exists(val_file_path):
        print(f"⚠️ 找不到验证集文件: {val_file_path}，跳过评估。")
        return 0.0

    val_prompts = []
    val_gts = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            p = data.get("prompt") or data.get("problem")
            gt = data.get("ground_truth") or data.get("expected_answer")
            val_prompts.append(p)
            val_gts.append(gt)

    print(f"🚀 正在使用 vLLM 进行极速评估 ({len(val_prompts)} 条)...")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</answer>"]
    )

    outputs = llm.generate(val_prompts, sampling_params, use_tqdm=True)
    
    total_reward = 0.0
    for output, gt in zip(outputs, val_gts):
        generated_text = output.outputs[0].text
        if "</answer>" not in generated_text:
            generated_text += "</answer>"
            
        result = reward_fn(generated_text, gt)
        total_reward += result.get("answer_reward", 0.0)

    accuracy = (total_reward / len(val_prompts)) * 100
    print(f"🎯 验证集准确率: {accuracy:.2f}%")
    return accuracy


# ================= 2. 主训练循环 =================

def train(args):
    # 配置实验名称
    run_name = f"sft-size-{args.limit if args.limit else 'full'}"
    wandb.init(project="cs336-sft-reasoning", name=run_name, config=vars(args))
    
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    print("加载 Tokenizer 和 模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0"
    )
    
    print("初始化 vLLM 引擎 (单卡模式，占用 20% 显存)...")
    llm = init_vllm(args.model_id, seed=42, gpu_memory_utilization=0.2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 加载数据
    prompts, responses = load_my_sft_data(args.train_data, limit=args.limit)

    # 评估初始模型 (Baseline)
    print("📊 训练前 Baseline 评估:")
    load_policy_into_vllm_instance(model, llm)
    baseline_acc = run_vllm_evaluation(llm, args.val_data, r1_zero_reward_fn)
    wandb.log({"eval/accuracy": baseline_acc, "eval_step": 0})

    global_train_step = 0
    eval_step = 1

    print("🔥 开始训练...")
    for epoch in range(args.epochs):
        model.train()
        
        # 将数据切成 Batch
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_responses = responses[i : i + args.batch_size]
            
            if len(batch_prompts) < args.micro_batch_size:
                continue
                
            optimizer.zero_grad()
            current_accum_steps = math.ceil(len(batch_prompts) / args.micro_batch_size)
            
            # --- 梯度累积循环 (Microbatches) ---
            for j in range(0, len(batch_prompts), args.micro_batch_size):
                mb_prompts = batch_prompts[j : j + args.micro_batch_size]
                mb_responses = batch_responses[j : j + args.micro_batch_size]
                
                # 1. 数据处理
                data = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids = data["input_ids"].to("cuda:0")
                labels = data["labels"].to("cuda:0")
                mask = data["response_mask"].to("cuda:0")
                
                # 2. 前向传播算概率
                outputs = get_response_log_probs(model, input_ids, labels)
                
                # 3. 反向传播 (内含 backward)
                loss, meta = sft_microbatch_train_step(
                    outputs["log_probs"], mask, current_accum_steps
                )
            
            # 4. 更新参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_train_step += 1
            
            wandb.log({
                "train/loss": loss.item(), 
                "train/avg_log_probs": meta.get("avg_log_probs", 0),
                "train_step": global_train_step
            })

            # ================= 4. 周期性评估 =================
            if global_train_step % args.eval_every == 0:
                print(f"\n[{global_train_step} 步] 同步权重并进行 vLLM 评估...")
                model.eval()
                load_policy_into_vllm_instance(model, llm)
                acc = run_vllm_evaluation(llm, args.val_data, r1_zero_reward_fn)
                
                wandb.log({"eval/accuracy": acc, "eval_step": eval_step})
                eval_step += 1
                model.train()

    # 训练结束后做最后一次评估
    print("🏁 训练结束，进行最终评估...")
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    final_acc = run_vllm_evaluation(llm, args.val_data, r1_zero_reward_fn)
    wandb.log({"eval/accuracy": final_acc, "eval_step": eval_step})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Assignment 5 SFT Training")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str, default="data/MATH/sft_full.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/val.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="限制训练数据的数量，比如 128, 256")
    parser.add_argument("--batch_size", type=int, default=32, help="全局有效 Batch Size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="每次前向传播的样本数 (防 OOM)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=50, help="每多少个 global step 评估一次")
    
    args = parser.parse_args()
    train(args)