from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from hf_gpt_model import GMQConfig, GMQModel

def get_prompts():
    prompt_datas = [
        '椭圆和圆的区别',
        '中国关于马克思主义基本原理',
        '人类大脑的主要功能是',
        '万有引力是',
        '世界上人口最多的国家是',
        'DNA的全称是',
        '数学中π的值大约是',
        '世界上最高的山峰是',
        '太阳系中最大的行星是',
        '二氧化碳的化学分子式是',
        '地球上最大的动物是',
        '地球自转一圈大约需要',
        '杭州市的美食有',
        '江苏省的最好的大学',
    ]
    return prompt_datas

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)
    
    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_file = "/root/autodl-tmp/gpt2/results_truncate/checkpoint-178829"
    model_file = "/root/autodl-tmp/llm/results_truncate/gmq_pretrain_truncate"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_file)
    model.to(device)
    model.eval()
    eval_prompts = get_prompts()

    with torch.no_grad():
        for prompt in eval_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt",return_attention_mask=True).to(device)

            output = model.generate(
                input_ids=input_ids['input_ids'],
                max_length = 100,
                temperature = 1,
                attention_mask = input_ids['attention_mask'],
                top_k = 50,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample = True
            )
            decode_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print(decode_text)