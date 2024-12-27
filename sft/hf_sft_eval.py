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
        "所有的猫都是动物。所有的动物都是猫。这两句话是一样的吗",
        "近日，某公司宣布推出新款智能手机，该手机采用了全新的设计语言，具备强大的性能和长续航。请总结这句话。"
        "春风绿江南岸。生成下联。",
        "苹果公司今天发布了新的iPhone 15。找出其中的实体。",
        "人工智能正在改变世界，特别是在医疗和教育领域。关键词是什么？",
        "我喜欢打蓝球和足求。纠正其中的错误。",
        "这部电影让我非常失望，剧情很平庸。情感是积极的还是消极的还是中性的？",
        "生成一个关于智能音箱的文案。"
        "如果下雨天你没有带伞，应该怎么办？需要你一步一步的思考。",
        "地球的公转周期是多少天？"
        "古诗仿写：窗含西岭千秋雪，门泊东吴万里船。",
        "第一句：我喜欢吃苹果。第二句：苹果是我最喜欢的水果。请问相似度是多少？"
        "青春，生成歌词。",
        "太阳系中有八大行星，地球是唯一已知存在生命的行星。地球在太阳系中有什么特点？",
        "学而时习之，不亦说乎？翻译文言文",
        "科技改变生活，生成作文",
        "张无忌站在山巅，远眺天边云海，心中百感交集。金庸风格续写。",
        "请用简短的方式介绍一下自己。"
    ]
    return prompt_datas

default_system ="你是“小辣”，一个友好的智能助手！"


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

