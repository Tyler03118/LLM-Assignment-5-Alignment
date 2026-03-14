import json
import os
import sys
from datasets import load_dataset

# 确保当前目录在路径中，以便导入项目模块
sys.path.append(os.getcwd())

try:
    # 导入作业提供的打分函数
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("❌ 错误: 无法导入 r1_zero_reward_fn。请确保在项目根目录下运行此脚本。")
    sys.exit(1)

def prepare_data():
    print("🚀 正在从 Hugging Face 下载官方 SFT Reasoning 数据集...")
    
    # 1. 加载包含推理轨迹的全量数据集 (Experiment 1 基础)
    # 根据 README，文件名是 sft_gpt-oss-120b.jsonl
    dataset = load_dataset(
        "garg-aayush/sft-cs336-assign5-datasets", 
        data_files="sft-reason/sft_gpt-oss-120b.jsonl", 
        split="train"
    )
    
    full_data = []
    filtered_data = []
    
    print(f"✅ 成功加载全量数据，共 {len(dataset)} 条。")
    print("开始执行 Experiment 2 的正确性过滤 (此过程涉及 LaTeX 解析，请稍候)...")

    for i, row in enumerate(dataset):
        # 提取字段
        problem = row.get("problem")
        trace = row.get("reasoning_trace")
        answer = row.get("extracted_answer")
        gt = row.get("expected_answer")
        
        # 💥 构造作业要求的标准 Response 格式
        # 这个格式必须与你之后的 tokenize_prompt_and_output 逻辑保持一致
        formatted_response = f"<think>\n{trace}\n</think> <answer> {answer} </answer>"
        
        item = {
            "prompt": problem,
            "response": formatted_response,
            "ground_truth": gt
        }
        full_data.append(item)
        
        # ======== 核心过滤逻辑 (Experiment 2) ========
        # 使用 r1_zero_reward_fn 判定该推理轨迹是否导向了正确答案
        reward_info = r1_zero_reward_fn(formatted_response, gt)
        
        # 只要 answer_reward == 1.0，说明模型这一题做对了
        if reward_info.get("answer_reward", 0.0) > 0:
            filtered_data.append(item)
        
        if (i + 1) % 500 == 0:
            print(f"已处理 {i + 1} / {len(dataset)} 条...")

    # 3. 保存文件
    output_dir = "data/MATH"
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, "sft_full.jsonl")
    filtered_path = os.path.join(output_dir, "sft_filtered.jsonl")

    with open(full_path, "w", encoding="utf-8") as f:
        for item in full_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(filtered_path, "w", encoding="utf-8") as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 4. 输出实验报告所需数据
    print("\n" + "="*40)
    print("📊 数据准备实验报告摘要 (SFT Experiment)")
    print("="*40)
    print(f"1. 原始全量数据集大小 (Full Dataset): {len(full_data)}")
    print(f"2. 过滤后正确数据集大小 (Filtered Dataset): {len(filtered_data)}")
    
    accuracy = (len(filtered_data) / len(full_data)) * 100
    print(f"3. 教师模型 (GPT-OSS-120B) 在此训练集上的准确率: {accuracy:.2f}%")
    print(f"4. 文件已保存至: {output_dir}")
    print("="*40)

if __name__ == "__main__":
    prepare_data()