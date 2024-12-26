from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig
from model.hf_lmq_model import LMQModel,LMQConfig
from utilities import check_network_params


def init_model_and_tokenizer(model_path):
    m_cfg, m_model, m_type = LMQConfig, LMQModel, LMQConfig.model_type
    AutoConfig.register(m_type, m_cfg)
    AutoModel.register(m_cfg, m_model)
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "dtype") and isinstance(config.dtype, str):
        config.dtype = getattr(torch, config.dtype, torch.float32)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModel.from_pretrained(model_path, config=config)
    model.to(device)
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



def get_prompts():
    prompt_datas = [
        "给周杰伦的<枫>写音乐评论。",
        "给王心凌的<爱你>写音乐评论。"
    ]
    return prompt_datas

default_system ="你是“小辣”，一个友好的智能助手，擅长处理以下任务：自然语言推理、文本摘要、对联生成、音乐评论、实体识别、关键词识别、文本纠错、情感分析、文案生成、链式思考、开放问答、古诗仿写、文本相似度匹配、歌词生成、阅读理解、文言文翻译、作文生成、金庸风格续写、自我介绍。请根据用户的输入内容，理解任务并提供相应的回答。"


def generate_template(system, prompt):
    if not system:
        system = default_system
    template = f"<|im_start|>system\n{system}<|im_end|>"
    template += f"\n<|im_start|>user\n{prompt}<|im_end|>"
    template += f"\n<|im_start|>assistant\n"
    return template

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # device = torch.device('cpu')
    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)

    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_file = "./results_sft/checkpoint-34362"
    # model_file = "./results_sft/gmq_sft"
    # model_name = "Qwen/Qwen2.5-0.5B"
    model_file = "./results/lma_sft"
    model, tokenizer = init_model_and_tokenizer(model_file)
    model.eval()
    # check_network_params(model)
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
                max_length=200,
                temperature=0.9,
                top_k=30,
                # top_p=0.9,
                attention_mask = input_ids['attention_mask'],
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            decode_text = tokenizer.decode(output[0], skip_special_tokens=False)
            print("---------------------------------------")
            print(decode_text)

