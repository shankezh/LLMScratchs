import json
from dataclasses import dataclass, asdict
from typing import List
from datasets import load_dataset

#############################
# For ShareGPT
############################
class Message:
    def __init__(self, _from:str, value:str):
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
            "conversations": [ msg.to_dict()  for msg in self.conversations],
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
                    sft_item = SFTItem(conversations=[Message("human",user),Message("gpt",assistant)], system=system, tools=[]).to_dict()
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
                sft_item = SFTItem(conversations=[Message("human",user),Message("gpt",assistant)], system=system, tools=[]).to_dict()
                fj.write(json.dumps(sft_item, ensure_ascii=False, indent=4))
                fj.write(",\n")
        fj.write("]")
        print("done..")


###############################################
# To get all category name with number of items
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
    with open("sft_data_meta.json", "w") as fw:
        category_dict["total_num"] = total
        fw.write(json.dumps(category_dict, ensure_ascii=False, indent=4))
    print("done..")


#######################################
# To split data with different sub-categories
#######################################
def split_datasets(path):
    name_list = ["NLI", "Summary", "OpenQA", "Cot", "MRC", "SentimentAnalyze", "ClassicalChinese", "AncientPoem", "BELLE"]


if __name__ == '__main__':
    # build_data("firefly-train-1.1M.jsonl")
    # build_cn_data("Summary", "sft_data_general.json")
    get_category_infos("sft_data_cn.json")
