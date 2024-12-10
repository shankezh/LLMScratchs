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
        '文本：联合国秘书长安东尼奥·古特雷斯在周一的一次演讲中强调了全球合作的重要性，并呼吁各国政府加强对气候变化的关注。要求：识别文本中提到的实体。',
        '文本：苹果公司在本周发布了其最新款智能手机iPhone 15，这款设备将在美国、欧洲和亚洲市场同步上市。要求：提取文本中的组织名称和地点。',
        '文本：《三体》这本书由刘慈欣创作，讲述了一个关于人类与外星文明接触的故事。要求：提取文本中的人名和书名。',
        '文本：北京冬奥会于2022年在中国北京举办，这一盛事吸引了来自全球200多个国家的运动员参加。要求：识别文本中的地点、事件名称以及数量相关信息。',
        '文本：SpaceX的猎鹰9号火箭在美国佛罗里达州成功发射，运载了一颗由NASA研发的气象卫星。要求：提取文本中的组织名称、地点和发射设备名称。',
        '文本：阿里巴巴创始人马云在一次公开演讲中表示，未来五年内电子商务行业将迎来快速增长。要求：提取文本中的人名、组织名称和时间。',
        '文本：新冠病毒疫苗由辉瑞公司和BioNTech联合研发，于2020年底获得美国食品药品监督管理局（FDA）的紧急使用授权。要求：识别文本中的组织名称、时间和事件名称。',
        '文本：乔布斯（Steve Jobs）在2007年推出了第一代iPhone，彻底改变了移动设备的行业格局。要求：提取文本中的人名、年份和产品名称。',
        '文本：中国长江三峡水电站是目前世界上最大的水电站，位于中国湖北省，是中国水利工程的杰出代表。要求：识别文本中的地点名称和工程名称。',
        '文本：2023年，特斯拉（Tesla）的市值突破一万亿美元，成为全球最有价值的汽车制造商之一。要求：提取文本中的年份、组织名称和数值相关信息。'
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
    model_file = "../sft/results_sft/gmq_sft_scene_NER"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_file)
    model.to(device)
    model.eval()
    # check_network_params(model)
    eval_prompts = get_prompts()

    with torch.no_grad():
        for idx, prompt in enumerate(eval_prompts):
            sampling_prompt = generate_template(pre_sys + task_descriptions['NER'], prompt)
            input_ids = tokenizer(sampling_prompt, return_tensors="pt", return_attention_mask=True).to(device)
            # 采样次数
            sampling_num = 10
            with open("sampling_verify.txt", "a") as f:
                f.write(f"==================={idx} start ==================\n")
                f.write(f"Q: {prompt}\n")
                for i in range(sampling_num):
                    output = model.generate(
                        input_ids=input_ids['input_ids'],
                        max_length=200,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        attention_mask=input_ids['attention_mask'],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                    )
                    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    f.write('A[' + str(i+1) + ']: ' + decoded_text.split("assistant\n", 1)[-1] + '\n')
                f.write(f"==================={idx} done ==================\n")
                f.close()
            print(f"the {idx + 1} / {len(eval_prompts)} is done ..")


        # prompt = eval_prompts[3]
        # sampling_prompt = generate_template(pre_sys + task_descriptions['NER'], prompt)
        # # sampling_prompt = generate_template(default_system, prompt)
        # input_ids = tokenizer(sampling_prompt, return_tensors="pt", return_attention_mask=True).to(device)
        #
        # # 采样次数
        # sampling_num = 20
        # sampling_len_list = []
        # sampling_appear_list = []
        # for i in range(sampling_num):
        #     output = model.generate(
        #         input_ids=input_ids['input_ids'],
        #         max_length=200,
        #         temperature=0.7,
        #         top_k=50,
        #         top_p=0.95,
        #         attention_mask=input_ids['attention_mask'],
        #         pad_token_id=tokenizer.pad_token_id,
        #         eos_token_id=tokenizer.eos_token_id,
        #         do_sample=True,
        #     )
        #     decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
        #     sampling_len_list.append(len(decoded_text))
        #     print(f"-------------count: {i+1}： {len(decoded_text)}-----------")
        #     print(decoded_text)
        #     # if len(decoded_text) == 221:
        #         # print(decoded_text)
        # print(sampling_len_list)