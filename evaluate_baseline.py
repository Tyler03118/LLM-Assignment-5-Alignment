import json
import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import List, Callable, Dict

# 导入作业提供的奖励函数（前提是你已克隆仓库）
# 注意：如果该函数仅适配 MATH，我们可能需要对 GSM8K 做轻微适配
try:
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    # 如果导入失败，定义一个基础的解析逻辑作为备选
    # 注意：签名需与真实函数保持一致 (response, ground_truth, fast=True)
    def r1_zero_reward_fn(response: str, ground_truth: str, fast: bool = True) -> Dict[str, float]:
        # 严格要求 "</think> <answer>...</answer>" 格式（与 grader 一致）
        has_format = "</think> <answer>" in response and "</answer>" in response
        if has_format:
            match = re.search(r'<answer>(.*?)<\/answer>', response, re.DOTALL)
            answer = match.group(1).strip() if match else ""
            is_correct = 1.0 if answer == ground_truth.strip() else 0.0
            return {"format_reward": 1.0, "answer_reward": is_correct, "reward": is_correct}
        else:
            return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}

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
        
        # 计算评估指标
        metrics = reward_fn(generated_text, gold)
        total_reward += metrics["reward"]
        
        # 构造序列化条目
        serialized_results.append({
            "prompt": prompts[i],
            "gold_answer": gold,
            "model_generation": generated_text,
            "metrics": metrics
        })
    
    # 将结果序列化到磁盘 
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in serialized_results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"评估完成。结果已保存至: {output_path}")
    print(f"平均准确率: {total_reward / len(prompts):.2%}")

if __name__ == "__main__":
    # 配置模型和采样参数
    MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,      
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 初始化 vLLM
    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.85)

    # 2. 加载数据 (用 GSM8K 替代 MATH)
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # 提取 GSM8K 的标准答案（通常在 #### 之后）
    def parse_gsm8k_gold(text):
        return text.split("####")[-1].strip()

    # 3. 格式化为 r1_zero prompt
    # 我们直接从仓库读取 prompt 文件以保证准确性
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 使用 replace 而非 str.format()，防止题目中出现 { } 导致崩溃
    prompts = [prompt_template.replace("{question}", item["question"]) for item in dataset]
    gold_answers = [parse_gsm8k_gold(item["answer"]) for item in dataset]

    # 4. 运行评估并保存结果
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=sampling_params,
        output_path="baseline_gsm8k_results.jsonl"
    )