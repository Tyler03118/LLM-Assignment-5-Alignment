import os
import json
import math
import argparse
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入作业要求的核心组件
from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ================= 1. 评估辅助函数 =================

def run_hf_evaluation(model, tokenizer, val_file_path, reward_fn, val_limit=100):
    """
    根据讲义要求，测量模型在 MATH 验证集上的性能
    """
    if not os.path.exists(val_file_path):
        print(f"⚠️ 找不到验证集文件: {val_file_path}，跳过评估。")
        return 0.0

    val_prompts = []
    val_gts = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f.readlines()[:val_limit]:
            data = json.loads(line)
            # 兼容不同的数据格式键名
            p = data.get("prompt") or data.get("problem")
            gt = data.get("ground_truth") or data.get("expected_answer")
            val_prompts.append(p)
            val_gts.append(gt)

    model.eval()
    total_reward = 0.0
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left" # 评估生成建议使用左填充

    batch_size = 4
    for i in tqdm(range(0, len(val_prompts), batch_size), desc="评估中"):
        batch_p = val_prompts[i:i + batch_size]
        batch_gt = val_gts[i:i + batch_size]

        inputs = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024, # 讲义要求的生成长度
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for j, output_ids in enumerate(outputs):
            generated_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            # 补全标签以适配评分器
            if "</answer>" not in generated_text:
                generated_text += "</answer>"
            # 使用官方提供的 r1_zero_reward_fn
            result = reward_fn(generated_text, batch_gt[j])
            total_reward += result.get("answer_reward", 0.0)

    tokenizer.padding_side = original_padding_side
    accuracy = (total_reward / len(val_prompts)) * 100
    return accuracy

# ================= 2. 主训练逻辑 =================

def train(args):
    run_name = f"sft-filtered-{args.learning_rate}"
    wandb.init(project="cs336-sft-reasoning", name=run_name, config=vars(args))
    
    # 按照讲义建议设置 wandb 指标
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    output_base_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16, # 讲义推荐使用 bfloat16 以节省内存
    ).to("cuda:0")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 加载数据（你现在使用的是过滤后的数据集）
    with open(args.train_data, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if args.limit: lines = lines[:args.limit]
        dataset = [json.loads(l) for l in lines]
    
    prompts = [d["prompt"] for d in dataset]
    responses = [d["response"] for d in dataset]
    print(f"✅ 数据加载成功: {len(prompts)} 条训练样本。")

    # --- 1. Baseline 评估 (训练前) ---
    print("\n📊 正在进行 Baseline 评估 (Step 0)...")
    baseline_acc = run_hf_evaluation(model, tokenizer, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
    print(f"🎯 Baseline 准确率: {baseline_acc:.2f}%")
    wandb.log({"eval/accuracy": baseline_acc, "eval_step": 0})

    global_train_step = 0
    eval_step = 1

    # --- 2. 训练循环 ---
    print("\n🔥 开始训练循环...")
    for epoch in range(args.epochs):
        model.train()
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_p = prompts[i : i + args.batch_size]
            batch_r = responses[i : i + args.batch_size]
            if len(batch_p) < args.micro_batch_size: continue
                
            optimizer.zero_grad()
            # 计算梯度累加步数以支持更大的有效 Batch Size
            current_accum_steps = math.ceil(len(batch_p) / args.micro_batch_size)
            
            for j in range(0, len(batch_p), args.micro_batch_size):
                mb_p = batch_p[j : j + args.micro_batch_size]
                mb_r = batch_r[j : j + args.micro_batch_size]
                
                # 构造响应掩码，仅对回复部分计算 Loss
                data = tokenize_prompt_and_output(mb_p, mb_r, tokenizer)
                input_ids = data["input_ids"].to("cuda:0")
                labels = data["labels"].to("cuda:0")
                mask = data["response_mask"].to("cuda:0")
                
                outputs = get_response_log_probs(model, input_ids, labels)
                loss, meta = sft_microbatch_train_step(outputs["log_probs"], mask, current_accum_steps)
            
            # 使用梯度裁剪 (Clip Grad Norm) 保持训练稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_train_step += 1
            
            wandb.log({
                "train/loss": loss.item(), 
                "train/avg_log_probs": meta.get("avg_log_probs", 0), 
                "train_step": global_train_step
            })

            # --- 3. 定期评估与保存 ---
            if global_train_step % args.save_every == 0:
                print(f"\n[{global_train_step} 步] 触发定期评估...")
                acc = run_hf_evaluation(model, tokenizer, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
                print(f"🎯 步数 {global_train_step} 准确率: {acc:.2f}%")
                wandb.log({"eval/accuracy": acc, "eval_step": eval_step})
                
                # 保存检查点
                ckpt_path = os.path.join(output_base_dir, f"checkpoint-{global_train_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                eval_step += 1
                model.train()

    # --- 4. 最终评估 (训练后) ---
    print("\n🏁 训练结束，正在进行最终评估...")
    final_acc = run_hf_evaluation(model, tokenizer, args.val_data, r1_zero_reward_fn, val_limit=args.val_limit)
    print(f"🏆 最终验证集准确率: {final_acc:.2f}%")
    wandb.log({"eval/accuracy": final_acc, "eval_step": eval_step})

    # --- 5. 最终模型保存 ---
    final_model_path = os.path.join(output_base_dir, "final_model")
    print(f"💾 正在保存最终权重至: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("✅ 任务圆满完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Assignment 5 SFT Complete Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data", type=str, default="data/MATH/sft_filtered.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--val_limit", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    
    train(parser.parse_args())