# Introduction
这里展示了如何构造SFT数据，由于本项目的预训练使用的数据集只有中文，因此无法形成多语言通用能力，且数据集各种类规模不一致，非常小的数据集可能会严重泛化表现，因此只提取部分数据。

# 数据处理
源数据：
[Firefly 流萤](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) <br>

位于当前目录下的sft_data_builder.py文件，通过get_category_infos函数检查数据类别和数量,得到信息如下：
```python
...
if __name__ == '__main__':
    get_category_infos("../data/firefly-train-1.1M.jsonl")
```
```shell
python sft_data_builder.py
```
等待命令行处理数据完毕，可以看到sft_data_meta.json文件在同级目录，内部显示了种类及数量.
```json
{
    "NLI": 50000,
    "Summary": 50000,
    "Couplet": 50000,
    "MusicComment": 50000,
    "NER": 50000,
    "KeywordRecognition": 50000,
    "TextCorrection": 50000,
    "SentimentAnalyze": 50000,
    "ProductDesc": 70000,
    "Cot": 74771,
    "OpenQA": 69843,
    "StoryGeneration": 19048,
    "AncientPoem": 69950,
    "TextMatching": 50000,
    "LyricGeneration": 49985,
    "MRC": 70000,
    "ClassicalChinese": 50000,
    "Composition": 50000,
    "Dictionary": 30895,
    "JinYongGeneration": 49990,
    "Translation": 50000,
    "ProseGeneration": 658,
    "Program": 974,
    "BELLE": 543285,
    "total_num": 1649399
}
```
初步选取只超过50,000条item的类别，且剔除英文翻译：<br>
NLI:自然语言推理,Summary: 总结摘要, Couplet:对联, MusicComment:音乐评论 <br>
NER: 实体识别, KeywordRecognition:关键词识别, TextCorrection:文本纠错<br>
SentimentAnalyze：情感分析, ProductDesc:文案生成,Cot: 链式思考, OpenQA:开放问答<br>
AncientPoem: 古诗仿写, TextMatching：文本相似度匹配, LyricGeneration:歌词生成<br>
MRC:阅读理解,ClassicalChinese:文言文翻译,Composition:作文生成,JinYongGeneration：金庸风格续写<br>
BELLE:通用指令

其中Cot是英文内容，BELLE包含了英文任务，需要做特殊处理：
这里的做法是把Cot翻译成中文






