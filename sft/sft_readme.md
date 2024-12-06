# Introduction
../sft
../sft/hf_sft.py  基于Huggingface的相关API实现的SFT训练代码，为什么不使用SFT_Trainer, 因为SFT_Trainer只支持全量加载数据，考虑到有时候数据集可能较大，因此依旧使用了Trainer+流式数据的形式，但是原理是一样的，没区别.<br>
../sft/hf_sft_eval.py 评估代码，由于设置了CPU处理，因此可以一边使用GPU训练，一边选择自己的模型进行效果评估展示.<br>

# DataFormat
这里使用了sharegpt风格的数据进行处理，所以请将数据格式转换成对应形式，否则请自行实现数据处理部分.<br>
数据样例[LLaMA-Factory sharegpt中文样例形式](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_zh_demo.json)

# Experiments
1. 对于数据集，对每个批次取其中长度最大的tokens的值，将当前批次的数据集填充至等长后进行训练，这种策略会加快训练速度，考虑到不等长的数据情况。<br>
2. 词嵌入模型使用了QWEN2.5，各种语言都支持很好，直接省事。<br>


# How-to-Start-SFT
```python
'python hf_sft.py'

# How-to-Eval.
>python hf_sft_eval.py
note: 请注意修改对应的权重文件的位置.
