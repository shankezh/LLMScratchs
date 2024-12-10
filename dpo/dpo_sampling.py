from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig
from utilities import check_network_params


def get_prompts():
    prompt_datas = [
        '什么是量子力学的基本原理？',
        '地球上有哪些重要的生态系统？',
        '写一个关于‘健康饮食’的短促销文案。',
        '以下两段话是否表达相同的意思？\n1. 我喜欢阅读。\n2. 阅读让我感到快乐。',
        '以一个迷失的探险家为主角，写一个惊险的短篇故事。',
        '以‘春天’为主题，写一首五言绝句。',
        '写一个 Python 程序，计算一个列表中所有元素的平均值。',
        '从以下句子中提取出人名和地名：\n‘爱丽丝在巴黎度过了一个美好的假期。’',
        '分析以下评论的情感倾向（正面/中性/负面）：\n‘这款手机性能非常强大，但电池续航让我失望。',
        '请说明‘一箭双雕’的含义。',
        '将以下英文句子翻译成中文：\n‘The weather today is sunny and pleasant.’',
        '为《射雕英雄传》中的郭靖设计一个新的冒险情节。',
        '给出以下逻辑题的详细推理过程：\n‘如果今天是周五，那么三天后是周几？’'
    ]
    return prompt_datas

default_system = r'你的名字是良智,是一个擅长回答问题的AI助手,请一步步地思考然后再帮助用户回答问题.'

def generate_template(system, prompt):
    if not system:
        system = default_system
    template = f"<|im_start|>system\n{system}<|im_end|>"
    template += f"\n<|im_start|>user\n{prompt}<|im_end|>"
    template += f"\n<|im_start|>assistant\n"
    return template

if __name__ == '__main__':
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)

    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_file = "./results_sft/checkpoint-34362"
    model_file = "../sft/results_sft/gmq_sft"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_file)
    model.to(device)
    model.eval()
    # check_network_params(model)
    eval_prompts = get_prompts()

    with torch.no_grad():
        prompt = eval_prompts[0]
        sampling_prompt = generate_template(default_system, prompt)
        input_ids = tokenizer(sampling_prompt, return_tensors="pt", return_attention_mask=True).to(device)

        # 采样次数
        sampling_num = 100
        sampling_len_list = []
        sampling_appear_list = []
        for i in range(sampling_num):
            output = model.generate(
                input_ids=input_ids['input_ids'],
                max_length=512,
                temperature=1.0,
                top_k=10,
                top_p=0.8,
                attention_mask=input_ids['attention_mask'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
            sampling_len_list.append(len(decoded_text))
            # print(f"-------------count: {i+1}： {len(decoded_text)}-----------")
            # print(decoded_text)
            if len(decoded_text) == 221:
                print(decoded_text)
        print(sampling_len_list)