from faker import Faker
import random
import json
from dataclasses import dataclass, asdict
from typing import List



# print(random.choice(name_builder).name())


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

def build_template(name, job):
    # 增加更多的 system 模板
    system_templates = [
        f"你叫{name}，是一个友好的{job}！",
        f"作为{name}，你是一位专业的{job}，随时准备帮助他人。",
        f"大家都称你为{name}，你是一位知识渊博的{job}。",
        f"你的名字是{name}，是一位热情的{job}。",
        f"你好，我的助手{name}，你是一位擅长{job}的专业人员。"
    ]

    # 用户输入模板
    user_forms = [
        "你叫什么名字？",
        "请问你的名字是什么？",
        "告诉我你的名字。",
        "你是谁？",
        "可以介绍一下你自己吗？",
        "你的名字叫什么？",
        "请自我介绍一下。"
    ]

    # 助手回答模板
    name_forms = [
        f"我叫{name}，是一个{job}。",
        f"我的名字是{name}，我是一位{job}。",
        f"我是{name}，我的工作是{job}。",
        f"大家都叫我{name}，我是一位从事{job}的助手。",
        f"您好！我是{name}，一名{job}。",
        f"名字是{name}，职业是{job}。"
    ]

    # 随机选择模板
    system_template = random.choice(system_templates)
    user = random.choice(user_forms)
    assistant = random.choice(name_forms)

    return system_template, user, assistant

if __name__ == '__main__':
    num_items = 29000
    name_builder = [Faker(locale='zh_CN'), Faker(locale='en_GB')]
    with open("sft_self_introduction.json", "w", encoding="utf-8") as f:
        for i in range(num_items):
            system, user, assistant = build_template(random.choice(name_builder).name(), random.choice(name_builder).job())
            sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)], system=system,
                               tools=[]).to_dict()
            json.dump(sft_item, f, ensure_ascii=False)
            f.write("\n")

        for i in range(1000):
            system, user, assistant = build_template("小辣", "AI助理!")
            sft_item = SFTItem(conversations=[Message("human", user), Message("gpt", assistant)], system=system,
                               tools=[]).to_dict()
            json.dump(sft_item, f, ensure_ascii=False)
            if i != 999:
                f.write("\n")
