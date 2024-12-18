# LLMScratchs
这个仓库用来反应关于大模型相关的学习成果，并且将会覆盖训练和推理两个部分。
对于代码部分，这个仓库会首先基于原理和相关学习资料，基于pytorch原生代码进行实现，然后再会根据当前热门框架，例如Huggingface等进行实现等。
在实现过程中，会顺手将一些收集到的调参，训练心得等学习资料进行整理，用于分享和记录。

包含训练方式：<br>
Pretrain guide:[Pretrain Link](pretrain) <br>
SFT guide:[SFT Link](sft) <br>
DPO guide:[DPO Link](dpo) <br>

目前已实现方式：<br>
Torch[√] <br>
HuggingFace[√]<br>
DeepSpeed[√] (一机多卡)<br>

目前包含模型：<br>
GPT 2<br>
LLAMA 2<br>
LLAMA 3.2<br>


## 推理
如果只有cpu，推荐:llama.cpp (如果使用知识简单使用开原模型，推荐安装ollama)
如果是GPU，推荐vLLM


## 模型和数据下载网站
> -[Huggingface](https://huggingface.co/models) <br>
> -[ModelScope](https://modelscope.cn)<br>

## Prompt工程相关
> 高质量提示词范例，可以考虑参考这个写自己的提示词或者做自己sft的数据处理相关的提示词工程：<br>
> https://github.com/langgptai/LangGPT<br>
> [LangGPT在飞书的页面，prompt各种模版](https://langgptai.feishu.cn/wiki/S8ZmwcW3aishrAklZMDcpBsmndb)<br>


## 大模型榜单相关
|网站名称|连接|榜单内容|
|:----:|:----:|:----:|
|数据学习|[link](https://www.datalearner.com/ai-models/leaderboard/datalearner-llm-leaderboard)|全球LLM大模型榜单|
|HuggingFace|[link](https://huggingface.co/spaces/mteb/leaderboard)|词嵌入模型榜单|
|HuggingFace|[link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)|OpenLLM榜单|
|HuggingFace|[link](https://huggingface.co/spaces?sort=trending&search=benchmark)|一些Benchmarks|

# Refs & Recommandations
ps：如果你对大模型底层技术感兴趣，并且十分想自己实现一次，我一定要推荐你去看这个仓库，从底层原理到代码实现全部包含：<br>
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)<br>
[1] [LLM训练实战经验](https://techdiylife.github.io/big-model-training/deepspeed/LLM-state-of-GPT.html#%E9%97%AE%E9%A2%981gpt%E6%A8%A1%E5%9E%8B%E6%98%AF%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83%E7%9A%84) <br>
[2][大模型训练与使用技巧](https://techdiylife.github.io/big-model-training/deepspeed/LLM-state-of-GPT.html#%E9%97%AE%E9%A2%981gpt%E6%A8%A1%E5%9E%8B%E6%98%AF%E5%A6%82%E4%BD%95%E8%AE%AD%E7%BB%83%E7%9A%84)<br>
