import os
import json
import math
import argparse
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入核心组件
from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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


def run_hf_evaluation(model, tokenizer, val_file_path, reward_fn, val_limit=100):
    if not os.path.exists(val_file_path):
        print(f"⚠️ 找不到验证集文件: {val_file_path}，跳过评估。")
        return 0.0

    val_prompts = []
    val_gts = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[:val_limit]:
            data = json.loads(line)
            p = data.get("prompt") or data.get("problem")
            gt = data.get("ground_truth") or data.get("expected_answer")
            val_prompts.append(p)
            val_gts.append(gt)

    print(f"🚀 正在使用原生 HF 评估 {len(val_prompts)} 条数据...")

    model.eval()
    total_reward = 0.0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_size = 4
    for i in tqdm(range(0, len(val_prompts), batch_size), desc="评估中"):
        batch_p = val_prompts[i:i + batch_size]
        batch_gt = val_gts[i:i + batch_size]

        inputs = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for j, output_ids in enumerate(outputs):
            generated_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            if "</answer>" not in generated_text:
                generated_text += "</answer>"
            result = reward_fn(generated_text, batch_gt[j])
            total_reward += result.get("answer_reward", 0.0)

    tokenizer.padding_side = original_padding_side
    accuracy = (total_reward / len(val_prompts)) * 100
    print(f"🎯 验证集准确率: {accuracy:.2f}%")
    return accuracy

# ================= 2. 主训练循环 =================

def train(args):
    run_name = f"sft-filtered-{args.learning_rate}"
    wandb.init(project="cs336-sft-reasoning", name=run_name, config=vars(args))
    
    # 路径准备
    output_base_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("加载 Tokenizer 和 模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 加载数据
    prompts, responses = load_my_sft_data(args.train_data, limit=args.limit)

    global_train_step = 0
    eval_step = 1

    print("🔥 开始训练...")
    for epoch in range(args.epochs):
        model.train()
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_responses = responses[i : i + args.batch_size]
            if len(batch_prompts) < args.micro_batch_size: continue
                
            optimizer.zero_grad()
            current_accum_steps = math.ceil(len(batch_prompts) / args.micro_batch_size)
            
            for j in range(0, len(batch_prompts), args.micro_batch_size):
                mb_prompts = batch_prompts[j : j + args.micro_batch_size]
                mb_responses = batch_responses[j : j + args.micro_batch_size]
                data = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids = data["input_ids"].to("cuda:0")
                labels = data["labels"].to("cuda:0")
                mask = data["response_mask"].to("cuda:0")
                outputs = get_response_log_probs(model, input_ids, labels)
                loss, meta = sft_microbatch_train_step(outputs["log_probs"], mask, current_accum_steps)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_train_step += 1
            
            wandb.log({"train/loss": loss.item(), "train/avg_log_probs": meta.get("avg_log_probs", 0), "train_step": global_train_step})

            # --- 定期评估与 Checkpoint 保存 ---
            if global_train_step % args.save_every == 0:
                print(f"\n[{global_train_step} 步] 进行评估与保存...")
                acc = run_hf_evaluation(model, tokenizer, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
                wandb.log({"eval/accuracy": acc, "eval_step": eval_step})
                
                ckpt_path = os.path.join(output_base_dir, f"checkpoint-{global_train_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"💾 Checkpoint 已保存至: {ckpt_path}")
                
                eval_step += 1
                model.train()

    # --- 最终保存 ---
    final_path = os.path.join(output_base_dir, "final_model")
    print(f"🏁 训练结束，正在保存最终模型到 {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("✅ 所有保存操作已完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 SFT with Checkpointing")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str, default="data/MATH/sft_filtered.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=120, help="每隔多少步保存一次 Checkpoint")
    
    args = parser.parse_args()
    train(args)