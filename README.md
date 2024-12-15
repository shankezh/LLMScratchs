# LLMScratchs
这个仓库用来反应关于大模型相关的学习成果，并且将会覆盖训练和推理两个部分。
对于代码部分，这个仓库会首先基于原理和相关学习资料，基于pytorch原生代码进行实现，然后再会根据当前热门框架，例如Huggingface等进行实现等。
在实现过程中，会顺手将一些收集到的调参，训练心得等学习资料进行整理，用于分享和记录。

包含训练方式：
Pretrain guide:[Pretrain Link](pretrain) <br>
SFT guide:[SFT Link](sft) <br>
DPO guide:[DPO Link](dpo) <br>

目前已实现方式：
Torch[√]
HuggingFace[√]
DeepSpeed[√] (一机多卡)

目前包含模型：
GPT 2
LLAMA 2
LLAMA 3.2

# Refs & Recommandations
ps：如果你对大模型底层技术感兴趣，并且十分想自己实现一次，我一定要推荐你去看这个仓库，从底层原理到代码实现全部包含：<br>
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)<br>
[1] [LLM训练实战经验](https://techdiylife.github.io/big-model-training/deepspeed/LLM-state-of-GPT.html#%E9%97%AE%E9%A2%981gpt%E6%A8%A1%E5%9E%8B%E6%98%AF%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83%E7%9A%84) <br>
