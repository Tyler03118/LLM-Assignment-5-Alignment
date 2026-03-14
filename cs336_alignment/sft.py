
import torch
import torch.nn.functional as F
import numpy as np

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    分别对 prompt 和 output 进行编码，拼接后生成 input_ids, labels 和 response_mask。
    """
    all_input_ids = []
    all_labels = []
    all_response_masks = []
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # 1. 分别编码（不加 padding，不加特殊开始符防止重复）
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        
        # 完整的序列
        full_ids = prompt_ids + output_ids
        
        # 2. 构造原始 mask (1 表示 response, 0 表示 prompt)
        # 长度与 full_ids 一致
        mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        
        all_input_ids.append(torch.tensor(full_ids))
        all_labels.append(torch.tensor(full_ids)) # labels 初始与 input_ids 一致
        all_response_masks.append(torch.tensor(mask))

    # 3. 填充 (Padding) 到当前 batch 的最大长度
    # 使用 pad_sequence，padding 值通常设为 tokenizer.pad_token_id
    # 注意：labels 的 padding 必须设为 -100，以便在 CrossEntropy 中被忽略
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        all_labels, batch_first=True, padding_value=-100
    )
    padded_masks = torch.nn.utils.rnn.pad_sequence(
        all_response_masks, batch_first=True, padding_value=0
    )

    # 4. 执行作业要求的切片 (Slicing) 与位移 (Shifting)
    # input_ids: 切掉最后一个 token (max_len - 1)
    input_ids = padded_input_ids[:, :-1]
    
    # labels: 切掉第一个 token (用于预测下一个 token)
    labels = padded_labels[:, 1:]
    
    # response_mask: 需要与 labels 对齐，所以也切掉第一个位置
    response_mask = padded_masks[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算每个 token 预测的熵（在词表维度 vocab_size 上计算）。
    
    Args:
        logits: 形状为 (batch_size, sequence_length, vocab_size) 的张量。
        
    Returns:
        形状为 (batch_size, sequence_length) 的张量，包含每个位置的熵。
    """
    # 1. 计算 log-probabilities (数值稳定)
    # log_p 形状: (batch_size, sequence_length, vocab_size)
    log_p = F.log_softmax(logits, dim=-1)
    
    # 2. 计算 probabilities
    # p 形状: (batch_size, sequence_length, vocab_size)
    p = F.softmax(logits, dim=-1)
    
    # 3. 计算熵 H = -sum(p * log_p)
    # 在词表维度 (-1) 上求和
    entropy = -torch.sum(p * log_p, dim=-1)
    
    return entropy



def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    获取每个 token 的条件 log-probabilities，并可选地获取熵。
    """
    # 1. 前向传播获取 logits
    # 形状: (batch_size, sequence_length, vocab_size)
    logits = model(input_ids).logits

    # 2. 计算数值稳定的 log-probabilities
    # 形状: (batch_size, sequence_length, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)

    # 3. 提取 labels 对应位置的 log-probability
    # 注意：labels 中可能包含 -100 (ignore_index)，gather 不支持负数索引
    # 我们先将 -100 替换为 0，提取完后再通过 mask 抹除
    labels_for_gather = labels.clone()
    labels_for_gather[labels_for_gather == -100] = 0
    
    # 使用 gather 在词表维度 (-1) 提取数据
    # unsqueeze(-1) 把 labels 从 (B, L) 变成 (B, L, 1)
    # gather 后结果是 (B, L, 1)，再 squeeze(-1) 回到 (B, L)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_for_gather.unsqueeze(-1)).squeeze(-1)

    # 4. 对被忽略的 token (label == -100) 进行掩码处理，将其 log_prob 设为 0
    log_probs = log_probs * (labels != -100).float()

    results = {"log_probs": log_probs}

    # 5. 可选：计算每一步的预测熵
    if return_token_entropy:
        # 使用上一节实现的 compute_entropy
        results["token_entropy"] = compute_entropy(logits)

    return results


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    对张量元素进行掩码求和，并除以一个常数进行归一化。
    """
    # 1. 确保 mask 和 tensor 类型一致 (float)
    # mask == 0 的位置对应的 tensor 元素会被设为 0
    masked_tensor = tensor * mask.to(tensor.dtype)
    
    # 2. 根据 dim 参数进行求和
    if dim is not None:
        summed = torch.sum(masked_tensor, dim=dim)
    else:
        # 如果 dim 为 None，则对整个张量所有元素求和
        summed = torch.sum(masked_tensor)
        
    # 3. 归一化：除以传入的常数（通常是 batch_size）
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在单个 microbatch 上执行前向和后向传播。
    """
    # 1. 计算负对数概率 (Negative Log-Likelihood)
    # 我们希望 log_p 越大越好，所以 Loss 是 -log_p
    nll_per_token = -policy_log_probs

    # 2. 调用之前的 helper 函数进行掩码求和并归一化
    # 这会得到当前 microbatch 的总 Loss
    loss = masked_normalize(
        tensor=nll_per_token,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None # 对整个 batch 进行全局求和
    )

    # 3. 为梯度累积调整 Loss
    # 这一步至关重要：backward 之前的缩放保证了梯度的平均值是正确的
    loss_to_backward = loss / gradient_accumulation_steps

    # 4. 执行反向传播
    # 计算模型参数的梯度并存储在 .grad 中
    loss_to_backward.backward()

    # 5. 准备元数据用于 Logging（可选，方便观察训练状态）
    # 注意：我们返回原始的 loss，而不是除以累积步数后的，这样 log 出来的数字才有物理意义
    metadata = {
        "loss": loss.item(),
        "avg_log_probs": (policy_log_probs * response_mask).sum().item() / response_mask.sum().item()
    }

    return loss, metadata


def log_generations(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: callable,
    max_new_tokens: int = 1024,
) -> dict:
    """
    在训练循环中记录模型的生成结果和关键指标。
    """
    # 1. 切换到评估模式，关闭梯度计算
    model.eval()
    
    logs = []
    all_entropies = []
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []

    with torch.no_grad():
        for prompt, gt in zip(prompts, ground_truths):
            # 将 prompt 转为 input_ids
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            # 2. 生成 Response，并且要求返回 Logits 以便计算熵
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                output_logits=True,
                return_dict_in_generate=True,
                do_sample=False  # 验证时通常使用 Greedy Search (贪心策略) 保证结果可复现
            )

            # 提取生成的 token (去除 prompt 部分)
            gen_seq = outputs.sequences[0, prompt_len:]
            response_str = tokenizer.decode(gen_seq, skip_special_tokens=True)
            seq_len = len(gen_seq)
            all_lengths.append(seq_len)

            # 3. 计算 Average Token Entropy
            # outputs.logits 是一个 tuple，包含了每一步生成的 logits
            if outputs.logits:
                # 将其堆叠成 (sequence_length, vocab_size) 的张量
                logits_tensor = torch.cat(outputs.logits, dim=0)
                # compute_entropy 期望的输入是三维 (batch, seq, vocab)，所以增加一个 batch 维度
                entropy = compute_entropy(logits_tensor.unsqueeze(0)).mean().item()
            else:
                entropy = 0.0
            all_entropies.append(entropy)

            # 4. 计算 Reward (调用你之前写的 r1_zero_reward_fn)
            reward_info = reward_fn(response_str, gt)
            
            # 5. 根据答案对错，分别统计长度
            if reward_info.get("answer_reward", 0.0) > 0:
                correct_lengths.append(seq_len)
            else:
                incorrect_lengths.append(seq_len)

            # 记录这条样本的完整信息
            logs.append({
                "prompt": prompt,
                "response": response_str,
                "ground_truth": gt,
                "format_reward": reward_info.get("format_reward", 0.0),
                "answer_reward": reward_info.get("answer_reward", 0.0),
                "total_reward": reward_info.get("reward", 0.0),
                "avg_token_entropy": entropy,
                "response_length": seq_len
            })

    # 6. 计算聚合的统计指标
    summary = {
        "avg_entropy": np.mean(all_entropies) if all_entropies else 0.0,
        "avg_length": np.mean(all_lengths) if all_lengths else 0.0,
        "avg_correct_length": np.mean(correct_lengths) if correct_lengths else 0.0,
        "avg_incorrect_length": np.mean(incorrect_lengths) if incorrect_lengths else 0.0,
        "generations": logs  # 具体的生成文本，后续可以喂给 wandb.Table 打印出来
    }

    # 恢复训练模式
    model.train()
    
    return summary