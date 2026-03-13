import json
import re
import os
import sys
from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import List, Callable, Dict

# --- 第一部分：确保能够导入仓库内的判分逻辑 ---
# 假设你在 /workspace/assignment5-alignment 目录下运行
sys.path.append(os.getcwd())

try:
    # 优先尝试从仓库自带的模块导入，这样能保证所有的数学处理逻辑（如 sympy）都是完整的
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    # 如果环境路径有问题，建议手动指定导入或检查目录结构
    print("错误：无法导入 cs336_alignment.drgrpo_grader。请确保你在仓库根目录下运行。")
    sys.exit(1)

# --- 第二部分：定义评估核心逻辑 ---
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    gold_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str
) -> None:
    """
    执行批量推理并保存结果。
    """
    print(f"开始推理 {len(prompts)} 条数据...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    serialized_results = []
    total_reward = 0.0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold = gold_answers[i]
        
        # 使用官方 reward_fn 进行判分
        metrics = reward_fn(generated_text, gold)
        total_reward += metrics.get("reward", 0.0)
        
        serialized_results.append({
            "prompt": prompts[i],
            "gold_answer": gold,
            "model_generation": generated_text,
            "metrics": metrics
        })
    
    # 写入结果
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in serialized_results:
            f.write(json.dumps(entry) + "\n")
            
    print(f"评估完成。结果已保存至: {output_path}")
    print(f"平均准确率 (MATH Baseline): {total_reward / len(prompts):.2%}")

# --- 第三部分：主程序执行 ---
if __name__ == "__main__":
    # 1. 初始化 vLLM 配置
    MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
    
    # 严格遵循作业要求的采样参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 在 A100 上建议预留一点显存给 vLLM 的缓存
    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.85)

    # 2. 加载 MATH 数据集, 镜像可能只有 train 分
    print("正在加载 MATH 数据集...")
    full_dataset = load_dataset("qwedsacf/competition_math", split="train")

    # 官方 MATH 测试集通常是 5000 条，我们可以取数据集的最后 5000 条
    # 或者为了快速跑通实验先取前 1000 条进行测试
    dataset = full_dataset.select(range(len(full_dataset) - 5000, len(full_dataset)))


    
    # 3. 读取并准备 Prompts
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    if not os.path.exists(prompt_path):
        print(f"错误：找不到 prompt 文件 {prompt_path}")
        sys.exit(1)
        
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # 针对 MATH 格式构建数据集
    prompts = [prompt_template.replace("{question}", item["problem"]) for item in dataset]
    gold_answers = [item["solution"] for item in dataset]

    # 4. 启动正式评估
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn, # 直接传递导入的官方函数
        prompts=prompts,
        gold_answers=gold_answers,
        eval_sampling_params=sampling_params,
        output_path="baseline_math_results.jsonl"
    )