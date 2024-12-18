# Introduction
如果有GPU，推荐使用VLLM。

# Installation
```shell
pip install vllm

# 如果考虑量化，例如 AWQ
pip install autoawq
```

# Example
这里演示使用Qwen2.5-32B-AWQ量化模型进行推理，运行示例代码自动下载模型，如果设备GPU不存在，可以先通过命令行下下来。
```shell
python vllm_qen25_32b.py
```
[option] 单独使用命令行下载模型：
```shell
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ --local-dir ./cache
```




# Refs
[1] [VLLM docs](https://docs.vllm.ai/en/latest/models/generative_models.html)<br>
[2] [QWEN docs](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)<br>
[3] [QWEN2.5-32B-AWQ HuggingFace Space](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)<br>
