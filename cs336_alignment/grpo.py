import torch
import numpy as np
from typing import Literal

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    计算每组 rollout 响应的奖励，并根据组大小进行规范化。
    """
    # 1. 计算原始奖励 (Raw Rewards), 首先遍历所有的 rollout_responses
    # 假设 reward_fn 返回一个 dict，我们需要提取其中的 "reward" 键
    all_raw_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        # eg. {"reward": 0.0, "format_reward": 0.1, "answer_reward": -0.1}
        score_dict = reward_fn(response, gt)
        # eg. [1.0, 0.0, 1.0] (这是一个包含 3 个浮点数的 List)
        all_raw_rewards.append(score_dict["reward"])
    
    # 转化为 Tensor 方便矩阵操作
    raw_rewards = torch.tensor(all_raw_rewards, dtype=torch.float32)
    
    # 2. 将奖励按组重塑 (Reshape: [num_groups, group_size])
    num_groups = len(raw_rewards) // group_size
    grouped_rewards = raw_rewards.view(num_groups, group_size)
    
    # 3. 计算组内统计量
    group_means = grouped_rewards.mean(dim=1, keepdim=True)  # [num_groups, 1]
    
    if normalize_by_std:
        # 标准 GRPO 逻辑 (公式 28)
        group_stds = grouped_rewards.std(dim=1, keepdim=True) # [num_groups, 1]
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        # Dr. GRPO 简化逻辑 (公式 31)
        advantages = grouped_rewards - group_means
    
    # 4. 展平回原始形状
    advantages = advantages.view(-1)
    
    # 5. 准备元数据 (用于监控训练状态)
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
    }
    
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个 token 的原生策略梯度损失。
    """
    # raw_rewards_or_advantages 形状: (batch_size, 1)
    # policy_log_probs 形状: (batch_size, sequence_length)
    
    # 根据公式 (32): loss = - Advantage * log_prob
    # PyTorch 会自动处理 (batch_size, 1) 对 (batch_size, seq_len) 的广播
    loss = -(raw_rewards_or_advantages * policy_log_probs)
    
    return loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    实现带 Clipping 的 GRPO 损失函数。
    """
    # 1. 计算重要性采样比率 (Ratio)
    # 注意：一定要用 log_probs 相减再取 exp，直接用 probs 相除会导致数值爆炸
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # 2. 准备两个 Surrogate (替代) 目标
    # 广播优点：advantages (batch, 1) 会自动对齐 policy_log_probs (batch, seq_len)
    surr1 = ratio * advantages
    
    # 使用 torch.clamp 进行剪切
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = clipped_ratio * advantages

    # 3. 计算最终 Loss (取 min 并加负号)
    # 注意：这里的 min 是逐元素对比
    per_token_loss = -torch.min(surr1, surr2)

    # 4. 收集元数据 (用于监控训练质量)
    # 看看有多少比例的 token 被“刹车”限制住了
    is_clipped = (surr2 < surr1).float() if torch.any(advantages > 0) else (surr2 > surr1).float()
    # 更严谨的写法是直接对比 surr1 和 surr2
    is_clipped = (surr2 != surr1).float() 
    
    metadata = {
        "clip_fraction": is_clipped.mean(),
        "mean_ratio": ratio.mean(),
        "max_ratio": ratio.max(),
    }

    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    策略梯度损失调度器，用于分发不同的损失计算逻辑。
    """
    metadata = {}

    if loss_type == "no_baseline":
        # 1. 基础检查：必须提供 raw_rewards
        assert raw_rewards is not None, "no_baseline 需要 raw_rewards"
        # 2. 直接调用 Naive 版本
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    elif loss_type == "reinforce_with_baseline":
        # 1. 基础检查：必须提供计算好的 advantages
        assert advantages is not None, "reinforce_with_baseline 需要 advantages"
        # 2. 同样调用 Naive 版本，只是传入的 A 变了
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    elif loss_type == "grpo_clip":
        # 1. 基础检查：参数必须全家桶
        assert advantages is not None and old_log_probs is not None and cliprange is not None, \
            "grpo_clip 需要 advantages, old_log_probs 和 cliprange"
        # 2. 调用带有 Clipping 逻辑的高级版
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    
    else:
        raise ValueError(f"未知的 loss_type: {loss_type}")

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    只对 mask==1 的位置求均值
    """
    # 先把 mask=0 的位置清零（排除在求和之外）
    masked_tensor = tensor * mask
    
    if dim is None:
        # 对所有 masked 元素求均值：总和 / mask为1的元素个数
        return masked_tensor.sum() / mask.sum()
    else:
        # 沿指定维度：每行(或列)的总和 / 该行mask为1的个数
        return masked_tensor.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行单个 micro-batch 的前向和后向传播。
    """
    # 1. 调用之前的 Wrapper 计算 per-token loss (batch, seq_len)
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 2. 在 Sequence 维度应用掩码平均 (得到每条回答的标量 loss)
    # 使用之前实现的 masked_mean，dim=1 表示对每个回答内部的 token 求平均
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # 3. 在 Batch 维度求平均，得到这个 micro-batch 的总 loss
    loss = per_example_loss.mean()

    # 4. 关键：针对梯度累积进行缩放
    # 因为 backward() 是累加梯度的，我们需要除以累积步数来获得正确的平均梯度
    loss_scaled = loss / gradient_accumulation_steps

    # 5. 执行反向传播
    loss_scaled.backward()

    # 返回未缩放的 loss 用于 logging（这样你在 WandB 看到的数值更直观）
    return loss, metadata