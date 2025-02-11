# Introduction
这里展示了如何构造SFT数据，由于本项目的预训练使用的数据集只有中文，因此无法形成多语言通用能力，且数据集各种类规模不一致，非常小的数据集可能会严重泛化表现，因此只提取部分数据。

# 数据处理
源数据：
[Firefly 流萤](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) <br>

反复使用sft_data_builder.py, 通过不同函数完成数据的构建，请注意不同的step需要单独运行，请自行按照顺序进行注释和反注释，同时个别步骤需要移动文件。
位于当前目录下的sft_data_builder.py文件，通过get_category_infos函数检查数据类别和数量,得到信息如下：
```
...
if __name__ == '__main__':
 
    # transfer data to ShareGPT format
    # build_data("firefly-train-1.1M.jsonl")

    # step 1: statistic all number of sub-class items
    # please move result file (sft_data_meta.json) to data floder
    get_category_infos("../data/firefly-train-1.1M.jsonl")

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

    # step 5: merge all sub-classes files to one jsonl file
    # merge_sft_data("./merged_sft_data.json", "./merged_sft_data_val.json", "./merged_sft_data_meta.json")
```
```shell
python sft_data_builder.py
# 运行完成后请拷贝sft_data_meta.json到data目录
# please move sft_data_meta.json to data folder
```
## 数据切分
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
>Note:其中Cot是英文内容，BELLE包含了英文任务，需要做特殊处理：
这里的做法是把Cot翻译成中文,使用大模型进行翻译，函数见sft_data_builder.py中的translate_cot_items().<br>
但是这样效率太低，因此为了掩饰演示，这里使用了其它的cot数据，供74770条数据.[ModelScope Link](https://modelscope.cn/datasets/YorickHe/CoT_zh/summary)<br>
转换成sharegpt格式见函数见sft_data_builder.py中的build_cot_cn_data()<br>
对于BELLE数据，直接通过抓关键词方法，剔除了英文翻译任务，函数见sft_data_builder.py中的delete_belle_eng()<br>

## 构造数据
本次训练的数据缺乏自我介绍类别，因此需要构造一些数据用于自我介绍:
```shell
python sft_data_generate.py
# 生成一个名为 "sft_data_self_introduction.json" 文件，包含了5000条自我介绍相关的简单数据。
```

因此正式用于SFT的数据是：


|Category| Number of Items | Commit  |
|:---|:----------------|:--------|
|NLI| 50000           | 自然语言推理  |
|Summary| 50000           | 总结摘要    |
|Couplet| 50000           | 对联      |
|MusicComment| 50000           | 音乐评论    |
|NER| 50000           | 实体识别    |
|KeywordRecognition| 50000           | 关键词识别   |
|TextCorrection| 50000           | 文本纠错    |
|SentimentAnalyze| 50000           | 情感分析    |
|ProductDesc| 70000           | 文案生成    |
|Cot| 74770           | 链式思考    |
|OpenQA| 69843           | 开放问答    |
|AncientPoem| 69950           | 古诗仿写    |
|TextMatching| 50000           | 文本相似度匹配 |
|LyricGeneration| 49985           | 歌词生成    |
|MRC| 70000           | 阅读理解    |
|ClassicalChinese| 50000           | 文言文翻译   |
|Composition| 50000           | 作文生成    |
|JinYongGeneration| 49990           | 金庸风格续写  |
|BELLE| 514193          | 通用指令    |
|Self-Introduction| 5000            | 自我介绍    |


```shell
# 使用merge_sft_data()函数,合并所有的类别文本，并且打乱顺序
python sft_data_builder.py
```


