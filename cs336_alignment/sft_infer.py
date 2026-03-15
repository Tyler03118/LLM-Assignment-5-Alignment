import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设置模型路径
# 根据你之前的训练日志，模型保存在这个目录下
model_path = "outputs/sft-filtered-2e-05/final_model"

print(f"正在从 {model_path} 加载模型...")

# 2. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # 建议使用与训练时相同的精度 [cite: 193]
    device_map="cuda:0"         # 指定使用第一块 GPU
)

# 3. 定义 r1_zero 提示词模板
# 讲义要求使用特定的模板来触发模型的推理能力 [cite: 78, 79, 80, 81, 82]
def format_prompt(question):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
        f"User: {question} Assistant: <think>"
    )

# 4. 准备你想测试的 prompt
test_prompts = [
    " Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "If x + y = 10 and x - y = 4, what is the value of x * y?",
    "Find the derivative of f(x) = x^2 + 3x + 5."
]

# 5. 开始推理
model.eval()
print("\n🚀 开始推理测试...\n")

for i, q in enumerate(test_prompts):
    full_prompt = format_prompt(q)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        # 设置生成参数，stop 字符串设为 </answer> 以符合讲义规范 [cite: 141, 144]
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码并提取生成的内容
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"--- 测试用例 {i+1} ---")
    print(f"问题: {q}")
    print(f"模型输出:\n<think>{response}\n")
    print("-" * 50)