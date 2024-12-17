# Introduction
LLM预训练的目的是为了实质上是为了让模型开始构造词和词之间的基本的维度距离，从表现上看，就是让大模型完成词语接龙，这些接龙从表面上看可能无意义，因为模型在训练阶段，是不会补充上下文环境关系，而后续的补充上下文关系则需要使用SFT去构造。
DeepSpeed使用方法:[DeepSpeedGuide](deep_speed_guide.md)


## 文件目录
> ./pretrain                        #当前目录<br>
> ./pretrain/hf_pretrain.py         #基于Huggingface，但上下文使用了拼接形式<br>
> ./pretrain/hf_pretrain_truncate.py    #基于Huggingface，但上下文使用了截断形式 <br>
> ./pretrain/hf_pretrain_truncate_eval.py # 基于Huggingface的API，测试文件<br>
> ./pretrain/torch_pretrain.py  # 基于原生pytorch <br>
> ./pretrain/deepspeed_pretrain.py # 基于DeepSpeed开发

## 数据格式
数据使用了jsonl的形式，每一行都是一条预训练数据,格式为{"Text":"content"}, 两条数据之间，使用特殊token <|endoftext|>进行分隔，这是GPT的方法，千问的训练也使用了这种方法，因此我也选择了这种，LLAMA使用的是<s></s>作为分隔标记，看个人喜好自己处理就好.<br>
最大长度可以根据你开放的上下文长度来定义，如果上下文长度不够，可以使用padding填充的方法，既末尾全部使用<|endoftext|>进行填充，但需要注意的是，至少要在末尾保留一个<|endoftext|>用来告诉模型，一段关联的上下文结束了，同时可以考虑生成attention_mask,这样可以在计算loss的时候，把多于的<|endoftext|>的填充标记位置忽略掉，加快计算效率.<br>
除了填充，也可以使用拼接的方法，这样上下文之间只需要使用<|endoftext|>分隔即可。

## 演示数据
预训练数据仓库，可以自己写个转换函数转换：[原始Data Link]<br>
ModelScope，已经转换为jsonl格式，会更方便: [ModelScope Link]<br>


# Using
## 使用方法
选择一种你喜欢的方式，单机训练，启动方法如下:
```shell
python hf_pretrain.py
```
or
```shell
python hf_pretrain_truncate.py
```
or
```shell
python torch_pretrain.py
```
多卡多机训练见上方DeepSpeed方案链接.

## 训练过程
这里使用了wandb进行展示.[Wandb Link](https://wandb.ai)
![img.png](imgs/pretrain_img.png)
可以看到两个批次效果如图.

## 接龙效果


# Experiments

