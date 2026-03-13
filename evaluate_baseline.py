import os
import json
import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import List, Callable, Dict

# 导入作业提供的奖励函数（前提是你已克隆仓库） [cite: 140]
# 注意：如果该函数仅适配 MATH，我们可能需要对 GSM8K 做轻微适配
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    # 如果导入失败，我们先定义一个基础的解析逻辑作为备选
    def r1_zero_reward_fn(model_output: str, ground_truth: str) -> Dict[str, float]:
        # 提取 <answer> 标签内容
        match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
        answer = match.group(1).strip() if match else model_output.strip()
        # 简单数值对比
        is_correct = 1.0 if answer == ground_truth else 0.0
        # 格式检查：是否包含闭合标签
        has_format = 1.0 if match else 0.0
        return {"reward": is_correct, "format_reward": has_format, "answer_reward": is_correct}

# 1. 按照作业要求定义 evaluate_vllm 接口 
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    gold_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str
) -> None:
    """
    评估模型，计算指标，并将结果序列化到磁盘 [cite: 157]。
    """
    # 执行批量推理 [cite: 147]
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    serialized_results = []
    total_reward = 0.0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold = gold_answers[i]
        
        # 计算评估指标 [cite: 148]
        metrics = reward_fn(generated_text, gold)
        total_reward += metrics["reward"]
        
        # 构造序列化条目 [cite: 148]
        serialized_results.append({
            "prompt": prompts[i],
            "gold_answer": gold,
            "model_generation": generated_text,
            "metrics": metrics
        })
    
    # 将结果序列化到磁盘 [cite: 148]
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in serialized_results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"评估完成。结果已保存至: {output_path}")
    print(f"平均准确率: {total_reward / len(prompts):.2%}")

if __name__ == "__main__":
    # 配置模型和采样参数 [cite: 141-144]
    MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
    
    sampling_params = SamplingParams(
        temperature=1.0, # [cite: 141]
        top_p=1.0,       # [cite: 141]
        max_tokens=1024, # [cite: 141]
        stop=["</answer>"], # [cite: 144]
        include_stop_str_in_output=True # [cite: 144]
    )

    # 初始化 vLLM
    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.85)

    # 2. 加载数据 (用 GSM8K 替代 MATH)
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # 提取 GSM8K 的标准答案（通常在 #### 之后）
    def parse_gsm8k_gold(text):
        return text.split("####")[-1].strip()

    # 3. 格式化为 r1_zero prompt [cite: 79-82, 147]
    # 我们直接从仓库读取 prompt 文件以保证准确性 [cite: 83]
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    prompts = [prompt_template.format(question=item["question"]) for item in dataset]
    gold_answers = [parse_gsm8k_gold(item["answer"]) for item in dataset]

    # 4. 运行评估并保存结果 [cite: 148]
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=sampling_params,
        output_path="baseline_gsm8k_results.jsonl"
    )