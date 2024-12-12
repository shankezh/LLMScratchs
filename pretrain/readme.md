# Introduction
LLM预训练的目的是为了实质上是为了让模型开始构造词和词之间的基本的维度距离，从表现上看，就是让大模型完成词语接龙，这些接龙从表面上看可能无意义，因为模型在训练阶段，是不会补充上下文环境关系，而后续的补充上下文关系则需要使用SFT去构造。

## 文件目录
> ./pretrain                        #当前目录<br>
> ./pretrain/hf_pretrain.py         #基于Huggingface，但上下文使用了拼接形式<br>
> ./pretrain/hf_pretrain_truncate.py    #基于Huggingface，但上下文使用了截断形式 <br>
> ./pretrain/hf_pretrain_truncate_eval.py # 基于Huggingface的API，测试文件<br>
> ./pretrain/torch_pretrain.py  # 基于原生pytorch <br>
