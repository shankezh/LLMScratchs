# Introduction
完成预训练阶段后，可以考虑先进行一个通用型的指令微调，然后在进行场景型的指令微调.
../sft
../sft/hf_sft.py  基于Huggingface的相关API实现的SFT训练代码，为什么不使用SFT_Trainer, 因为SFT_Trainer只支持全量加载数据，考虑到有时候数据集可能较大，因此依旧使用了Trainer+流式数据的形式，但是原理是一样的，没区别.<br>
../sft/hf_sft_eval.py 评估代码，由于设置了CPU处理，因此可以一边使用GPU训练，一边选择自己的模型进行效果评估展示.<br>

# DataFormat
这里使用了sharegpt风格的数据进行处理，所以请将数据格式转换成对应形式，否则请自行实现数据处理部分.<br>
数据样例[LLaMA-Factory sharegpt中文样例形式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_zh_demo.json)

# Experiments
1. 对于数据集，对每个批次取其中长度最大的tokens的值，将当前批次的数据集填充至等长后进行训练，这种策略会加快训练速度，考虑到不等长的数据情况。<br>
2. 词嵌入模型使用了QWEN2.5，各种语言都支持很好，直接省事。<br>
3. 什么时候用SFT:
  > Prompt过于复杂
  > 输出格式不能稳定满足要求
4. 如何害怕破坏模型预训练参数，可以使用LoRA技术微调 <br>
5. 针对不同场景下，数据量级的建议：
  > 文案生成，剧本创作，小说续写等生成类任务：2～3k。
  > 参考问答：2k ~ 1w。
  > 文本分类：1～3k，和类别数量以及任务难易度强相关，类别较多/任务较难的场景可能需要1w条以上
6. SFT学习率设计
  > 通用性指令 SFT：通常使用稍大的学习率（如 1e-5 或 2e-5），因为在这个阶段，模型需要学会各种基本的指令和任务。较大的学习率能够加快模型学习的速度。
  > 场景 SFT：这个阶段的学习率通常较小（如 1e-6 到 5e-6 或 3e-6 ~ 1e-6 ）之间，有时甚至更小。，因为此时微调的目标是让模型适应特定任务和场景，因此需要更精细地调整模型的权重。较小的学习率有助于避免过度拟合和破坏已学到的通用性知识。

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

