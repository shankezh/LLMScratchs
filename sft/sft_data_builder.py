import json
from dataclasses import dataclass, asdict
from typing import List
from datasets import load_dataset


class Message:
    def __init__(self, _from:str, value:str):
        self.__dict__["from"] = _from
        self.__dict__["value"] = value

    def to_dict(self):
        return self.__dict__


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


if __name__ == '__main__':
    build_data("firefly-train-1.1M.jsonl")
