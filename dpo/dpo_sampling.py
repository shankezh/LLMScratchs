from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig
from utilities import check_network_params

pre_sys = "你叫良智，是一名多功能的 AI 助手，当前的任务类型是："

task_descriptions = {
    "NLI": "自然语言推理。",
    "TextMatching": "文本匹配。",
    "StoryGeneration": "故事生成。",
    "ProductDesc": "商品文案生成。",
    "Summary": "文本摘要。",
    "AncientPoem": "古诗生成。",
    "NER": "实体识别。",
    "SentimentAnalyze": "情感分析。",
    "TextCorrection": "文本纠错。",
    "Couplet": "对对联。",
    "MusicComment": "生成音乐热评。",
    "KeywordRecognition": "关键词识别。",
    "ClassicalChinese": "翻译成文言文。",
    "Cot": "思维链思考。",
    "LyricGeneration": "歌词生成。",
    "Translation": "中英翻译。",
    "OpenQA": "开放问答。",
    "Composition": "作文生成。",
    "MRC": "阅读理解。",
    "JinYongGeneration": "金庸风格小说生成。",
    "Dictionary": "成语释义。",
}

def get_prompts():
    prompt_datas = [
        '实体识别： 同时，香港人也决心保证充分实施基本法。 抽取出文中所有的实体',
        '鹄峙鸾翔 这个成语的意思是什么？',
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
        sampling_prompt = generate_template(pre_sys + task_descriptions['NER'], prompt)
        # sampling_prompt = generate_template(default_system, prompt)
        input_ids = tokenizer(sampling_prompt, return_tensors="pt", return_attention_mask=True).to(device)

        # 采样次数
<<<<<<< HEAD
        sampling_num = 100
=======
        sampling_num = 10
>>>>>>> 8c290608e1091e17b67c68563300da8f9136b426
        sampling_len_list = []
        sampling_appear_list = []
        for i in range(sampling_num):
            output = model.generate(
                input_ids=input_ids['input_ids'],
<<<<<<< HEAD
                max_length=512,
                temperature=1.0,
                top_k=10,
                top_p=0.8,
=======
                max_length=200,
                temperature=0.7,
                top_k=10,
                top_p=0.95,
>>>>>>> 8c290608e1091e17b67c68563300da8f9136b426
                attention_mask=input_ids['attention_mask'],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
            decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
            sampling_len_list.append(len(decoded_text))
<<<<<<< HEAD
            # print(f"-------------count: {i+1}： {len(decoded_text)}-----------")
            # print(decoded_text)
            if len(decoded_text) == 221:
                print(decoded_text)
=======
            print(f"-------------count: {i+1}： {len(decoded_text)}-----------")
            print(decoded_text)
            # if len(decoded_text) == 221:
                # print(decoded_text)
>>>>>>> 8c290608e1091e17b67c68563300da8f9136b426
        print(sampling_len_list)