from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig
from utilities import check_network_params


def get_prompts():
    prompt_datas = [
        '你叫什么名字',
        '中国的首都是哪里？',
        '中国有哪些比较好的大学？',
        '全世界最好的大学是什么？',
        '怎样自学易经？',
        '你知道长江吗？',
        '人类的血液主要由哪些成分组成？',
        '第一颗人造卫星是哪个国家发射的？',
        '你知道杭州有什么美食吗？',
        '你知道泰山在哪里吗？',
        '地球上最大的动物是什么？',
        '地球自转一圈大约需要多少时间？',
        '人类最早使用的金属是什么？',
        '水的化学分子式是什么？',
        '大气层中含量最多的气体是什么？',
        '世界上最高的山峰是什么？',
        '你知道世界上最深的海沟是什么吗？',
        '最早发明印刷术的是哪个国家？',
        '万有引力是谁提出的？',
        '光合作用的主要原理是什么？',
        '你知道大熊猫的主要食物是什么吗？',
        '海水为什么是咸的？',
        '我们平时喝的牛奶主要含有什么营养成分？',
        '一星期有多少天？'
    ]
    return prompt_datas

default_system = r'你的名字是良智,你是一个擅长回答问题的人工智能助手.'

def generate_template(system, prompt):
    if not system:
        system = default_system
    template = f"<|im_start|>system\n{system}<|im_end|>"
    template += f"\n<|im_start|>user\n{prompt}<|im_end|>"
    template += f"\n<|im_start|>assistant\n"
    return template

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() e'lse cpu')
    device = torch.device('cpu')
    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)

    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_file = "./results_sft/checkpoint-43200"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_file)
    model.to(device)
    model.eval()
    check_network_params(model)
    eval_prompts = get_prompts()

    # for prompt in eval_prompts:
    #     print(generate_template(default_system, prompt))
    #     break

    with torch.no_grad():
        for prompt in eval_prompts:
            # print("Q:",prompt)
            sft_prompt = generate_template(default_system, prompt)
            # input_ids = tokenizer(sft_prompt, return_tensors="pt").input_ids.to(device)
            input_ids = tokenizer(sft_prompt, return_tensors="pt",return_attention_mask=True).to(device)
            # print("A:")
            # print(input_ids.shape)
            output = model.generate(
                input_ids=input_ids['input_ids'],
                max_length=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                attention_mask = input_ids['attention_mask'],
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,

                do_sample=True
            )
            decode_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print("---------------------------------------")
            print(decode_text)

