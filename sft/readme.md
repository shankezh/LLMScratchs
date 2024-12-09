# Introduction
完成预训练阶段后，可以考虑先进行一个通用型的指令微调，然后在进行场景型的指令微调.<br>

## 目录说明
```shell
../sft
../sft/hf_sft.py
../sft/hf_sft_eval.py
../sft/sft_data_builder.py
```
> hf_sft.py 基于Huggingface的相关API实现的SFT训练代码，为什么不使用SFT_Trainer, 因为SFT_Trainer只支持全量加载数据，考虑到有时候数据集可能较大，因此依旧使用了Trainer+流式数据的形式，但是原理是一样的，没区别.<br>
> hf_sft_eval.py 评估代码，由于设置了CPU处理，因此可以一边使用GPU训练，一边选择自己的模型进行效果评估展示.<br>
> sft_data_builder.py 用于转换FireFly数据集至ShareGPT格式,但转换后需要配合使用shell命令去除其中尾部的错误标志，shell命令见下方.<br>
  > ```shell
  > sed -i '$d' sft_data_general.json # 删除最后一行的"]"
  > sed -i '$d' sft_data_general.json # 继续删除最后一行的"},"
  >  sed -i '$ s/$/}]/' sft_data_general.json # 在最后一行添加"}]"
  >  ```
## 数据相关
本项目使用SFT数据集地址：
[Firefly 流萤](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) <br>

# DataFormat
## 训练数据
这里使用了sharegpt风格的数据进行处理，所以请将数据格式转换成对应形式，否则请自行实现数据处理部分.<br>
数据样例[LLaMA-Factory sharegpt中文样例形式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_zh_demo.json) <br>
## 处理形式
处理形式很多，这里借用了Qwen2.5的处理形式，这样写的好处是方便以后使用千问的基座模型做微调.
```shell
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
{assistant_prompt}<|im_end|>
```

# Experiments
1. 对于数据集，对每个批次取其中长度最大的tokens的值，将当前批次的数据集填充至等长后进行训练，这种策略会加快训练速度，考虑到不等长的数据情况。<br>
2. 词嵌入模型使用了QWEN2.5，各种语言都支持很好，直接省事。<br>
3. 什么时候用SFT:
  > Prompt过于复杂 <br>
  > 输出格式不能稳定满足要求<br>
4. 如何害怕破坏模型预训练参数，可以使用LoRA技术微调 <br>
5. 针对不同场景下，数据量级的建议：
  > 文案生成，剧本创作，小说续写等生成类任务：2～3k。<br>
  > 参考问答：2k ~ 1w。<br>
  > 文本分类：1～3k，和类别数量以及任务难易度强相关，类别较多/任务较难的场景可能需要1w条以上<br>
6. SFT学习率设计
  > 通用性指令 SFT：通常使用稍大的学习率（如 1e-5 或 2e-5），因为在这个阶段，模型需要学会各种基本的指令和任务。较大的学习率能够加快模型学习的速度。<br>
  > 场景 SFT：这个阶段的学习率通常较小（如 1e-6 到 5e-6 或 3e-6 ~ 1e-6 ）之间，有时甚至更小。，因为此时微调的目标是让模型适应特定任务和场景，因此需要更精细地调整模型的权重。较小的学习率有助于避免过度拟合和破坏已学到的通用性知识。<nr>
7. SFT的时候可以考虑带上15%-30%的通用指令，防止通用能力下降(可选). <br>
8. Epoch选择，通常2-5轮，但一定要注意看loss，生成类任务可以放多一点，5-10轮，但生成类loss没有意义，自己可以通过外部方案写评估策略. <br>

# How-to-Start-SFT
```shell
python hf_sft.py
```
# How-to-Eval.
```shell
python hf_sft_eval.py
```
note: 请注意修改对应的权重文件的位置.


# Refs
[1] [LLMs from Scratchs ch7] (https://github.com/rasbt/LLMs-from-scratch) <br>
[2] [火山方舟文档](https://www.volcengine.com/docs/82379/1221664#%E4%B8%80%E4%BA%9B%E5%BB%BA%E8%AE%AE) <br>
[3] [Qwen官方文档](https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html) <br>

