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
        "文本：联合国秘书长安东尼奥·古特雷斯在周一的一次演讲中强调了全球合作的重要性。要求：识别文本中的人名。",
        "文本：苹果公司发布了iPhone 15，这款设备将在全球市场同步上市。要求：提取文本中的组织名称。",
        "文本：《三体》这本书由刘慈欣创作，讲述了一个关于人类与外星文明接触的故事。要求：提取文本中的书名。",
        "文本：北京冬奥会于2022年在中国北京举办，吸引了来自200多个国家的运动员参加。要求：识别文本中的地点。",
        "文本：SpaceX的猎鹰9号火箭在佛罗里达州发射，运载了一颗气象卫星。要求：提取文本中的发射设备名称。",
        "文本：阿里巴巴创始人马云在一次演讲中表示，电子商务行业将迎来快速增长。要求：提取文本中的人名。",
        "文本：新冠病毒疫苗由辉瑞公司研发，于2020年底获得紧急使用授权。要求：识别文本中的年份。",
        "文本：乔布斯（Steve Jobs）在2007年推出了第一代iPhone，彻底改变了移动设备行业格局。要求：提取文本中的产品名称。",
        "文本：中国长江三峡水电站是世界上最大的水电站，位于湖北省。要求：识别文本中的工程名称。",
        "文本：2023年，特斯拉的市值突破一万亿美元。要求：提取文本中的组织名称。",
        "文本：马斯克在加州的一个会议上讨论了火星殖民的可能性。要求：识别文本中的人名。",
        "文本：谷歌公司在其最新的开发者大会上发布了一系列AI技术。要求：提取文本中的组织名称。",
        "文本：《百年孤独》是加西亚·马尔克斯的代表作之一，获得了诺贝尔文学奖。要求：提取文本中的书名。",
        "文本：东京奥运会在2021年举行，受到了全球观众的关注。要求：识别文本中的地点。",
        "文本：蓝色起源的新谢泼德火箭在德克萨斯州成功发射。要求：提取文本中的发射设备名称。",
        "文本：马化腾在一个业内论坛上讨论了互联网的未来发展方向。要求：提取文本中的人名。",
        "文本：COVID-19疫苗是由Moderna公司开发的，已经开始在全球范围内推广使用。要求：识别文本中的年份。",
        "文本：苹果公司的首席执行官蒂姆·库克在产品发布会上展示了新一代MacBook。要求：提取文本中的产品名称。",
        "文本：巴黎埃菲尔铁塔是法国著名的地标性建筑，每年吸引成千上万的游客。要求：识别文本中的工程名称。",
        "文本：微软公司在2024年的市值达到了新高。要求：提取文本中的组织名称。",
        "文本：国际货币基金组织(IMF)在其年度报告中预测全球经济增长。要求：提取文本中的组织名称。",
        "文本：法国作家雨果在19世纪创作了《悲惨世界》。要求：提取文本中的书名。",
        "文本：巴塞罗那奥林匹克运动会于1992年在西班牙举办。要求：识别文本中的地点。",
        "文本：亚马逊的无人机配送系统在加州首次进行了试飞。要求：提取文本中的产品名称。",
        "文本：脸书创始人马克·扎克伯格在社交媒体上发表了关于隐私政策的声明。要求：提取文本中的人名。",
        "文本：2021年科技公司Zoom因疫情期间的表现出色，市值大幅上涨。要求：识别文本中的年份。",
        "文本：苹果公司的新总部位于加州库比蒂诺市。要求：识别文本中的地点。",
        "文本：华为公司推出的Mate系列手机获得了市场的高度评价。要求：提取文本中的产品名称。",
        "文本：长城位于中国，是世界七大奇迹之一。要求：识别文本中的工程名称。",
        "文本：摩根大通银行在2022年的金融报告会上公布了年度业绩。要求：提取文本中的组织名称。",
        "文本：巴黎圣母院是法国著名的历史建筑，吸引了全球的游客。要求：识别文本中的地点名称。",
        "文本：特斯拉公司在2023年推出了其最新的电动车型Model S Plaid。要求：提取文本中的产品名称。",
        "文本：亚马逊雨林是地球上最大的热带雨林，位于南美洲。要求：识别文本中的地点。",
        "文本：诺贝尔和平奖由阿尔弗雷德·诺贝尔设立，旨在表彰对和平的贡献。要求：提取文本中的人名。",
        "文本：国际足联世界杯是全球最受欢迎的足球赛事，每四年举办一次。要求：识别文本中的事件名称。",
        "文本：IBM公司推出了其新一代量子计算机，标志着技术的一大步。要求：提取文本中的组织名称。",
        "文本：哈利波特系列书籍由J.K.罗琳创作，风靡全球。要求：提取文本中的书名。",
        "文本：耐克是一家总部位于美国俄勒冈州的全球运动品牌。要求：识别文本中的组织名称。",
        "文本：温哥华是加拿大的一个主要城市，以其自然景观和多元文化而闻名。要求：识别文本中的地点。",
        "文本：美国独立宣言在1776年由托马斯·杰弗逊撰写。要求：提取文本中的文档名称。",
        "文本：马斯克在Twitter上宣布SpaceX将发射首个去火星的宇航任务。要求：识别文本中的组织名称。",
        "文本：微软Surface是一系列由微软开发的个人计算器产品。要求：提取文本中的产品名称。",
        "文本：伦敦大本钟是英国著名的地标性建筑之一。要求：识别文本中的地点。",
        "文本：安格拉瓦特是柬埔寨的古老庙宇，吸引了数百万游客。要求：识别文本中的地点。",
        "文本：《星球大战》是由乔治·卢卡斯创作的著名科幻电影系列。要求：提取文本中的电影名称。",
        "文本：汉堡是德国第二大城市，位于易北河畔。要求：识别文本中的地点。",
        "文本：尼亚加拉大瀑布是北美洲最大的瀑布之一，跨越美国和加拿大。要求：识别文本中的地点。",
        "文本：玛雅文明是古代美洲的一个文化，以其雕塑和建筑闻名。要求：提取文本中的文明名称。",
        "文本：大英博物馆收藏了来自全球的数百万件艺术和历史展品。要求：提取文本中的机构名称。",
        "文本：谷歌地图是由谷歌公司开发的地图服务，提供卫星图片和街景功能。要求：识别文本中的产品名称。",
        "文本：奥普拉·温弗瑞是美国著名的电视主持人和慈善家。要求：识别文本中的人名。",
        "文本：梵高博物馆位于荷兰阿姆斯特丹，展示了梵高的许多著名画作。要求：提取文本中的地点名称。",
        "文本：澳大利亚大堡礁是世界上最大的珊瑚礁系统，由2900多个个别礁石组成。要求：识别文本中的地点。",
        "文本：白金汉宫是英国王室的主要官邸，位于伦敦。要求：提取文本中的地点名称。",
        "文本：尼尔·阿姆斯特朗是第一个登月的宇航员。要求：识别文本中的人名。",
        "文本：黄石国家公园是美国最古老的国家公园之一，以其地热活动而知名。要求：识别文本中的地点。",
        "文本：圣托里尼岛是希腊的一个岛屿，以其壮丽的日落和白色建筑而闻名。要求：提取文本中的地点名称。",
        "文本：墨西哥城是墨西哥的首都和最大城市，拥有丰富的文化和历史遗迹。要求：识别文本中的地点。",
        "文本：埃及金字塔是古埃及的象征，是世界七大奇迹之一。要求：提取文本中的地点名称。",
        "文本：巴塞罗那足球俱乐部是西班牙的一支著名足球队，拥有众多球迷。要求：识别文本中的组织名称。",
        "文本：耶鲁大学位于美国康涅狄格州，是一所享誉世界的顶尖学府。要求：提取文本中的地点名称。",
        "文本：比尔·盖茨是微软公司的创始人之一，也是一位著名的慈善家。要求：识别文本中的人名。",
        "文本：罗马斗兽场是古罗马时期的圆形剧场，至今仍吸引着大量游客。要求：提取文本中的地点名称。",
        "文本：亚洲开发银行致力于减少亚洲地区的贫困，总部设在菲律宾。要求：识别文本中的组织名称。",
        "文本：埃菲尔铁塔是巴黎的象征之一，由古斯塔夫·埃菲尔设计。要求：提取文本中的地点名称。",
        "文本：2024年夏季奥林匹克运动会将在巴黎举办。要求：识别文本中的事件名称。",
        "文本：马尔代夫以其清澈的海水和美丽的沙滩闻名于世。要求：提取文本中的地点名称。",
        "文本：达芬奇的《蒙娜丽莎》是世界著名的画作，现藏于巴黎卢浮宫。要求：识别文本中的艺术作品名称。",
        "文本：康奈尔大学是美国常春藤联盟学校之一，位于纽约州。要求：提取文本中的地点名称。",
        "文本：牛津大学是世界上最古老的英语大学之一，位于英国。要求：识别文本中的地点名称。",
        "文本：悉尼歌剧院是澳大利亚的标志性建筑，以其独特的帆船形状著称。要求：提取文本中的地点名称。",
        "文本：谷歌在全球范围内提供云计算服务，总部位于加利福尼亚州。要求：识别文本中的组织名称。",
        "文本：莫扎特是一位奥地利作曲家，对古典音乐有重大贡献。要求：提取文本中的人名。",
        "文本：杜拜塔是世界上最高的建筑，位于阿联酋杜拜。要求：识别文本中的地点名称。",
        "文本：安妮·弗兰克的日记是第二次世界大战期间的一个重要历史文献。要求：提取文本中的书名。",
        "文本：纽约时代广场是美国著名的旅游景点之一，位于曼哈顿。要求：识别文本中的地点名称。",
        "文本：威尼斯电影节是世界上历史最悠久的电影节之一，每年在意大利举办。要求：识别文本中的事件名称。",
        "文本：尼泊尔的喜马拉雅山是世界上最高的山脉，其中包括珠穆朗玛峰。要求：提取文本中的地点名称。",
        "文本：《时代》杂志是一本国际知名的周刊，涵盖广泛的政治和文化话题。要求：提取文本中的杂志名称。",
        "文本：自由女神像是美国的象征之一，位于纽约港口。要求：识别文本中的地点名称。",
        "文本：慕尼黑啤酒节是德国的传统节日，每年吸引成千上万的游客。要求：识别文本中的事件名称。",
        "文本：Facebook是由马克·扎克伯格在哈佛大学时创立的社交网络平台。要求：提取文本中的组织名称。",
        "文本：圣彼得大教堂位于梵蒂冈，是天主教的重要象征。要求：提取文本中的地点名称。",
        "文本：诺曼底登陆是第二次世界大战中的一个重要事件，标志着盟军反攻的开始。要求：识别文本中的事件名称。",
        "文本：安第斯山脉是南美洲的主要山脉，贯穿多个国家。要求：提取文本中的地点名称。",
        "文本：伦敦眼是英国的一个著名观光地标，提供壮观的城市全景。要求：识别文本中的地点名称。",
        "文本：普拉多博物馆位于西班牙马德里，收藏有许多欧洲艺术品。要求：提取文本中的机构名称。",
        "文本：埃弗雷斯特山是世界最高峰，位于尼泊尔和中国边界。要求：识别文本中的地点名称。",
        "文本：雷克雅未克是冰岛的首都，以其干净的空气和独特的北欧文化而闻名。要求：提取文本中的地点名称。",
        "文本：2024年东京奥运会将展示日本的文化和技术成就。要求：识别文本中的事件名称。",
        "文本：哥伦比亚大学位于纽约市，是一所著名的私立研究型大学。要求：识别文本中的地点名称。",
        "文本：《寻梦环游记》是皮克斯动画工作室制作的一部动画电影，深受小朋友和家庭的喜爱。要求：提取文本中的电影名称。",
        "文本：大熊猫是中国的国宝，主要生活在四川省的山区。要求：识别文本中的动物名称。",
        "文本：泰姬陵是印度的著名古迹，位于阿格拉市，是一座壮丽的陵墓。要求：提取文本中的地点名称。",
        "文本：梵高博物馆收藏了许多梵高的画作，位于荷兰阿姆斯特丹。要求：识别文本中的机构名称。",
        "文本：金门大桥是美国旧金山的标志性建筑，以其红色的桥体著称。要求：提取文本中的地点名称。",
        "文本：2025年将在洛杉矶举办夏季奥林匹克运动会。要求：识别文本中的事件名称。",
        "文本：乔丹是篮球历史上的传奇人物，曾效力于芝加哥公牛队。要求：提取文本中的人名。",
        "文本：泰晤士河流经英国伦敦，是城市的重要地理标志。要求：识别文本中的地点名称。",
        "文本：《哈利·波特》系列是J.K.罗琳创作的世界畅销书，深受读者喜爱。要求：提取文本中的书名。"
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
    # model_file = "../sft/results_sft/gmq_sft_scene_NER"
    model_file = "./results/gmq_dpo"
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