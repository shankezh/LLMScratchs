import json
from dataclasses import dataclass, asdict
from typing import List


#############################
# For ShareGPT
############################
class Message:
    def __init__(self, _from: str, value: str):
        self.__dict__["from"] = _from
        self.__dict__["value"] = value

    def to_dict(self):
        return self.__dict__


###################################
# For ShareGPT format
@dataclass
class SFTItem:
    conversations: List[Message]
    system: str
    tools: []

    def to_dict(self):
        return {
            "conversations": [msg.to_dict() for msg in self.conversations],
            "system": self.system,
            "tools": []
        }


###########################################################
# Extract specific one category from file
###########################################################
def build_specific_data(name, path):
    count = 0
    with open("sft_data_{name}.json", "a+") as fj:
        fj.write("[\n")
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                system = data["kind"]
                user = data["input"]
                assistant = data["target"]
                if system == "NER":
                    count += 1
                    print(f"{count} ..")
                    sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)], system=system,
                                       tools=[]).to_dict()
                    fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                    fj.write(",\n")
        fj.write("]")
        print("done..")


#############################################
# transfer data to shareGPT format
#############################################
def build_data(path):
    count = 0
    with open("sft_data_general.json", "a+") as fj:
        fj.write("[\n")
        with open(path, "r") as f:
            for line in f:
                count += 1
                print(f"{count}..")
                data = json.loads(line)
                system = data["kind"]
                user = data["input"]
                assistant = data["target"]
                sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)], system=system,
                                   tools=[]).to_dict()
                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                fj.write(",\n")
        fj.write("]")
        print("done..")


###############################################
# To get all categories name with number of items
###############################################
def get_category_infos(path):
    print("Building category infos...")
    category_dict = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["kind"] in category_dict.keys():
                # update number of item
                item_num = category_dict[data["kind"]]
                item_num += 1
                category_dict[data["kind"]] = item_num
            else:
                print("find the new one ...")
                # find the first one item for current category
                category_dict[data["kind"]] = 1
    # get total of items
    total = 0
    for category, item_num in category_dict.items():
        total += item_num
    with open("../data/sft_data_meta.json", "w") as fw:
        category_dict["total_num"] = total
        fw.write(json.dumps(category_dict, ensure_ascii=False, indent=4))
    print("done..")


#######################################
# To split data with different sub-categories
#######################################
def split_datasets(path):
    with open("../data/sft_data_meta.json", "r") as fj:
        meta = json.load(fj)
    name_list = ["NLI", "Summary", "Couplet", "MusicComment", "NER", "KeywordRecognition", "TextCorrection",
                 "SentimentAnalyze",
                 "ProductDesc", "Cot", "OpenQA", "AncientPoem", "TextMatching", "LyricGeneration", "MRC",
                 "ClassicalChinese",
                 "Composition", "JinYongGeneration", "BELLE"]
    name_dict = {name: 0 for name in name_list}

    import os
    os.makedirs("./data_subs", exist_ok=True)

    # open target file
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            system = data["kind"]
            user = data["input"]
            assistant = data["target"]
            if system in name_dict.keys() and system in meta.keys():
                name_dict[system] += 1
                print(f"{system} : {name_dict[system]}")
                with open(f"./data_subs/sft_data_{system}.json", "a+") as fj:
                    if name_dict[system] == 1:
                        fj.write("[\n")
                        sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                           system=system, tools=[]).to_dict()
                        fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                        fj.write(",\n")
                    elif name_dict[system] == meta[system]:
                        sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                           system=system, tools=[]).to_dict()
                        fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                        fj.write("]\n")
                    else:
                        sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                           system=system, tools=[]).to_dict()
                        fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                        fj.write(",\n")

                fj.close()
    print("done..")


########################################
# using LLM to translate items
#######################################
def translate_cot_items(path):
    from inference.vllm.vllm_example import init_llm_and_tokenizer, build_gpt_format, build_sampling_params
    import re

    with open("../data/sft_data_meta.json", "r") as fj:
        meta = json.load(fj)

    num_cot = meta["Cot"]

    model_path = "../inference/vllm/cache"
    model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    llm, tokenizer = init_llm_and_tokenizer(model_path=model_path, model_name=model_name, quantization="GPTQ",
                                            max_model_len=2048, dtype="auto")

    zero_shot_prompt = r'你是zero-shot COT任务翻译大师，当前需要做数据处理，需要对链式思考任务进行翻译，要求翻译链式思考相关输入任务中的英语，将其转换成中文，要求翻译合理，通顺，流程且意思不变，请注意，你只需要翻译input和output对应的部分不需要进行额外扩展或改写，务必使用中文。结果依旧使用"input-cn": "翻译内容" 和 "output-cn": "翻译内容" 进行输出.'

    # prompt = ("当前需要做数据处理，我需要对链式思考任务进行翻译，要求翻译链式思考相关输入任务中的英语，将其转换成中文，要求翻译合理，通顺，流程且意思不变，请注意，你只需要翻译input部分，但output部分如果你认为推理不合理或者不够细致，请你进行完善，但务必使用中文。结果依旧使用input-cn 和 output-cn进行标注.")

    with open(path, "r") as f:
        data = json.load(f)
        loss_items_idx = []
        batch_message = []
        batch_size = 5   # 50,000 items can be divide exactly
        batch_count = 0
        process_count = 1
        system = "Cot"
        sampling_params = build_sampling_params(max_tokens=2048)
        for idx, item in enumerate(data):
            conversations = item.get("conversations", [])
            human, gpt = None, None
            for conversation in conversations:
                if conversation.get("from") == "human":
                    human = conversation.get("value", "")
                elif conversation.get("from") == "gpt":
                    gpt = conversation.get("value", "")
                else:
                    raise ValueError(f"Unexpected 'from' value: {conversation.get('from')}")

            if not human or not gpt:
                raise ValueError(f"Missing 'human' or 'gpt' conversation at index {idx}")

            message = build_gpt_format(system=zero_shot_prompt, user=f"input:\n{human}\noutput:{gpt}")
            batch_message.append(message)
            batch_count += 1
            if batch_count == batch_size:
                outputs = llm.chat(messages=batch_message, sampling_params=sampling_params, use_tqdm=True)
                batch_message = []
                batch_count = 0
                for idx_out, output in enumerate(outputs):
                    translate_res = output.outputs[0].text
                    match = re.search(r"input-cn:\n(.*?)\noutput-cn:(.*)", translate_res, re.S)
                    if match:
                        user = match.group(1).strip()
                        assistant = match.group(2).strip()
                        print(f"Input-CN: {user}")
                        print(f"Output-CN: {assistant}")
                        print(f"------------------{idx+1}-------------------------------")
                        with open(f"./data_subs/sft_data_Cot_CN.json", "a+") as fj:
                            if process_count == 1:
                                fj.write("[\n")
                                sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                                   system=system, tools=[]).to_dict()
                                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                                fj.write(",\n")
                            elif process_count == num_cot:
                                sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                                   system=system, tools=[]).to_dict()
                                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                                fj.write("]\n")
                            else:
                                sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)],
                                                   system=system, tools=[]).to_dict()
                                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                                fj.write(",\n")
                            print(f"The {process_count} process is done.")
                            process_count += 1
                    else:
                        global_idx = idx - batch_size + idx_out
                        loss_items_idx.append(global_idx)
                        print(f"loss a match content: {global_idx} ...")
        with open("loss_item_cot.txt", "w") as fw:
            fw.write(str(loss_items_idx))

def build_cot_cn_data():
    import pandas as pd
    path = "../data/CoT_cn.csv"
    df = pd.read_csv(path)
    system = "CoT"
    m_len = df.__len__()
    with open("./data_subs/sft_data_Cot_CN.json", "a", encoding='utf-8') as fj:
        for i in range(m_len):
            if i == 0:
                fj.write("[")
                sft_item = SFTItem(conversations=[Message("human", df["instruction"].iloc[i]), Message("gpt", df["output"].iloc[i])],
                                   system=system, tools=[]).to_dict()
                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                fj.write(",\n")
            elif i == m_len - 1:
                sft_item = SFTItem(conversations=[Message("human", df["instruction"].iloc[i]), Message("gpt", df["output"].iloc[i])],
                                   system=system, tools=[]).to_dict()
                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                fj.write("]")
            else:
                sft_item = SFTItem(conversations=[Message("human", df["instruction"].iloc[i]), Message("gpt", df["output"].iloc[i])],
                                   system=system, tools=[]).to_dict()
                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                fj.write(",\n")
            print(f"The {i} process is done.")
        print("done ...")

####################################################
# delete english conversations in BELLE data
# tip: key word is "翻译"
def delete_belle_eng():
    print("start ...")
    count = 0
    key_words = "翻译"
    with open("./data_subs/sft_data_BELLE.json", "r", encoding='utf-8') as fj:
        data = json.load(fj)
        with open("./data_subs/sft_data_BELLE_CN.json", "a", encoding='utf-8') as fa:
            fa.write("[")
            data_len = len(data)
            for idx, item in enumerate(data):
                conversations = item.get("conversations", [])
                for conversation in conversations:
                    if conversation.get("from") == "human":
                        # find translate task
                        if key_words in conversation["value"]:
                            count = count + 1
                            print(f"find {count} items ...")
                            continue
                        else:
                            fa.write(json.dumps(item, ensure_ascii=False, indent=4))
                            if idx != (data_len-1):
                                fa.write(",\n")
            fa.write("]")
        print("done ...")


def merge_sft_data(output_file, val_file, meta_file):
    import os
    import random
    import json

    # 数据类别及其文件路径配置
    data_categories = {
        "NLI": "sft_data_NLI.json",
        "Summary": "sft_data_Summary.json",
        "Couplet": "sft_data_Couplet.json",
        # "MusicComment": "sft_data_MusicComment.json",
        "NER": "sft_data_NER.json",
        "KeywordRecognition": "sft_data_KeywordRecognition.json",
        "TextCorrection": "sft_data_TextCorrection.json",
        "SentimentAnalyze": "sft_data_SentimentAnalyze.json",
        "ProductDesc": "sft_data_ProductDesc.json",
        "Cot": "sft_data_Cot_CN.json",
        "OpenQA": "sft_data_OpenQA.json",
        "AncientPoem": "sft_data_AncientPoem.json",
        "TextMatching": "sft_data_TextMatching.json",
        "LyricGeneration": "sft_data_LyricGeneration.json",
        "MRC": "sft_data_MRC.json",
        "ClassicalChinese": "sft_data_ClassicalChinese.json",
        "Composition": "sft_data_Composition.json",
        "JinYongGeneration": "sft_data_JinYongGeneration.json",
        "BELLE": "sft_data_BELLE_CN.json",
        "Self-Introduction": "sft_data_self_introduction.json"
    }

    # 数据加载函数
    def load_data(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)  # 读取整个 JSON 数组

    # 加载所有数据
    data_pool = {}
    for category, file_path in data_categories.items():
        data_prefix = "./data_subs/"
        aim_path = data_prefix + file_path
        if os.path.exists(aim_path):
            print(aim_path)
            data_pool[category] = load_data(aim_path)
            print("done")
        else:
            raise  ValueError(f"Warning: File not found for category {category} at {aim_path}")

    # 初始化验证集和训练集
    val_data = []
    train_data = []

    # 从每个类别中挑选10条作为验证集
    for category, data in data_pool.items():
        random.shuffle(data)
        val_samples = data[:10]  # 前10条作为验证集
        train_samples = data[10:]  # 剩余数据作为训练集

        val_data.extend(val_samples)
        train_data.extend(train_samples)

    # 随机打乱训练数据
    def shuffle_data(data, seed=42):
        random.seed(seed)
        random.shuffle(data)

    shuffle_data(train_data)

    # 保存训练数据到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # 保存验证数据到文件
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    # 创建元数据文件
    meta_data = {
        "training_data_count": len(train_data),
        "validation_data_count": len(val_data)
    }

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    print(f"Data merged and saved to {output_file}")
    print(f"Validation data saved to {val_file}")
    print(f"Metadata saved to {meta_file}")

def build_simple_qa(path):
    with open(path, 'r', encoding='utf-8') as fr:
        with open("sft_train_data.json", 'w', encoding='utf-8') as fw:
            for idx, line in enumerate(fr):
                data = json.loads(line)
                user = data["question"]
                assistant = data["answer"]
                system = "simple_qa"
                sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)], system=system,
                                   tools=[]).to_dict()
                json.dump(sft_item, fw, ensure_ascii=False)
                fw.write("\n")
                print("idx",idx)


def shuffle_sft_data(path_list):
    import random
    train_save_path = "../data/sft_train_data.jsonl"
    val_save_path = "../data/sft_val_data.jsonl"
    meta_path = "../data/sft_meta_data.json"

    train_data = []
    val_data = []
    for path in path_list:
        current_data = []
        with open(path, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line != "":
                    current_data.append(json.loads(line))
        random.seed(42)
        random.shuffle(current_data)
        val_data.extend(current_data[:10])
        train_data.extend(current_data[10:])
    random.seed(42)
    random.shuffle(train_data)
    random.seed(42)
    random.shuffle(val_data)
    meta_data = {
        "training_data_count": len(train_data),
        "validation_data_count": len(val_data)
    }
    with open(train_save_path, 'w', encoding='utf-8') as f:
        for entry in train_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    with open(val_save_path, 'w', encoding='utf-8') as f:
        for entry in val_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':    # transfer data to ShareGPT format
    # build_data("firefly-train-1.1M.jsonl")

    # step 1: statistic all number of sub-class items
    # please move result file (sft_data_meta.json) to data floder
    # get_category_infos("../data/firefly-train-1.1M.jsonl")

    # step 2: split datasets to sub_datasets
    # will generate some files in the data_subs floder
    # split_datasets("../data/firefly-train-1.1M.jsonl")

    ## DEPRECATED, but you still can learn this way to build your data. translate COT datasets from English to Chinese by QWEN2.5
    # translate_cot_items("./data_subs/sft_data_Cot.json")

    # step 3: transfer COT datasets format to ShareGPT
    # note: please notice replace cot datasets, download link in the note part of "数据切分"
    # build_cot_cn_data()

    # step 4: Delete English content in Belle class
    # delete_belle_eng()

    # before start this step, please run sft_data_generate.py to create self-introduction.json
    # step 5: merge all sub-classes files to one jsonl file
    # merge_sft_data("../data/merged_sft_data.json", "../data/merged_sft_data_val.json", "../data/merged_sft_data_meta.json")


    # build_simple_qa("../data/train.jsonl")
    shuffle_sft_data(["../data/sft_train_sim_qa.json", "../data/sft_self_introduction.json"])