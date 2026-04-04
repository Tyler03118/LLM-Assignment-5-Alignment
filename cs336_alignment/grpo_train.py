import torch
import torch.nn.functional as F
import numpy as np
import typer
import random
from typing import Literal, Optional
from tqdm import tqdm
import wandb # 强烈建议用于 2x A100 的实验监控
import os
import contextlib
from cs336_alignment.grpo import compute_group_normalized_rewards, compute_naive_policy_gradient_loss, compute_grpo_clip_loss, compute_policy_gradient_loss, masked_mean, grpo_microbatch_train_step



# =========================================================================
# 第二部分：DDP 主训练循环 (The Engine)
# =========================================================================

app = typer.Typer()

@app.command()
def train_grpo(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
):
    """
    GRPO 分布式训练主循环。
    """
    # --------------------------------------------------------
    # 1. DDP 初始化
    # --------------------------------------------------------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = (global_rank == 0)

    # --------------------------------------------------------
    # 2. 全局到局部的 Batch Size 切分
    # --------------------------------------------------------
    # 基础完整性检查
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size 必须能被 gradient_accumulation_steps 整除"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size 必须能被 group_size 整除"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    # DDP 切分
    assert rollout_batch_size % world_size == 0, "rollout_batch_size 必须能被 GPU 数量整除"
    local_rollout_batch_size = rollout_batch_size // world_size
    local_n_prompts = n_prompts_per_rollout_batch // world_size
    local_train_batch_size = train_batch_size // world_size
    local_grad_accum_steps = gradient_accumulation_steps // world_size

    if is_main_process:
        print(f"🚀 初始化 DDP: World Size = {world_size}")
        print(f"📦 全局参数: Rollout={rollout_batch_size}, Accum_Steps={gradient_accumulation_steps}")
        print(f"🔪 局部参数 (Per GPU): Prompts={local_n_prompts}, Micro_Batch={micro_train_batch_size}, Local_Accum={local_grad_accum_steps}")

    # --------------------------------------------------------
    # 3. 模型与优化器初始化 (此处为占位，需替换为你自己的模型加载逻辑)
    # --------------------------------------------------------
    # model = AutoModelForCausalLM.from_pretrained("your-model-path").to(device)
    # model = DDP(model, device_ids=[local_rank])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    
    # 占位符，仅为了代码能通过语法检查运行
    class DummyModel(torch.nn.Module):
        def forward(self, *args, **kwargs): pass
    model = DDP(DummyModel().to(device), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ============================================================
    # 主循环开始
    # ============================================================
    for step in range(n_grpo_steps):
        if is_main_process:
            print(f"\n--- Step {step + 1}/{n_grpo_steps} ---")
        
        # --------------------------------------------------------
        # 阶段 A: Rollout (采样阶段)
        # --------------------------------------------------------
        # prompts = get_random_prompts(train_dataset, local_n_prompts)
        # repeated_prompts = [p for p in prompts for _ in range(group_size)]
        
        # model.eval()
        # with torch.no_grad():
        #     # 推荐使用 HF 的 generate，避免 vLLM 显存冲突
        #     # generated_texts = model.module.generate(...)
        pass
        
        # --------------------------------------------------------
        # 阶段 B: Reward 计算与优势值标准化
        # --------------------------------------------------------
        # raw_rewards_list = [reward_fn(text, gt)["reward"] for ...]
        
        # 模拟生成数据 (确保都在正确的 device 上)
        dummy_raw_rewards = torch.randn(local_n_prompts, group_size, device=device) 
        
        advantages = compute_group_normalized_rewards(
            dummy_raw_rewards, use_std_normalization, advantage_eps
        ).view(local_rollout_batch_size, 1)

        # --------------------------------------------------------
        # 阶段 C: 数据准备
        # --------------------------------------------------------
        # input_ids, labels, response_mask = tokenize_prompt_and_output(...)
        
        # --------------------------------------------------------
        # 阶段 D: 模型更新 (Optimizer Loop)
        # --------------------------------------------------------
        model.train()
        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(local_rollout_batch_size, device=device)
            epoch_loss = 0.0
            
            for i in range(0, local_rollout_batch_size, micro_train_batch_size):
                mb_indices = indices[i : i + micro_train_batch_size]
                is_last_step = (i + micro_train_batch_size >= local_rollout_batch_size)
                
                # mb_input_ids = input_ids[mb_indices]
                # mb_labels = labels[mb_indices]
                # mb_response_mask = response_mask[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # 模拟 forward pass 获取的数据
                mb_policy_log_probs = torch.randn(micro_train_batch_size, 1024, requires_grad=True, device=device)
                mb_response_mask = torch.ones(micro_train_batch_size, 1024, device=device)
                mb_old_log_probs = torch.randn(micro_train_batch_size, 1024, device=device) if loss_type == "grpo_clip" else None
                mb_raw_rewards = dummy_raw_rewards.view(-1, 1)[mb_indices] if loss_type == "no_baseline" else None

                # 核心魔法：只有在最后一步才同步梯度，否则暂停同步加速训练
                sync_context = model.no_sync() if not is_last_step else contextlib.nullcontext()
                
                with sync_context:
                    # 获取真实的 log_probs (需手动替换)
                    # current_results = get_response_log_probs(model, mb_input_ids, mb_labels)
                    # mb_policy_log_probs = current_results["log_probs"]
                    
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=mb_policy_log_probs,
                        response_mask=mb_response_mask,
                        gradient_accumulation_steps=local_grad_accum_steps,
                        loss_type=loss_type,
                        raw_rewards=mb_raw_rewards,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        cliprange=cliprange
                    )
                    epoch_loss += loss.item()

            # 梯度剪切与步进
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if is_main_process:
                print(f"Epoch {epoch+1} 结束, Loss: {epoch_loss / local_grad_accum_steps:.4f}")

if __name__ == "__main__":
    app()