# Introduction
如果有GPU，推荐使用VLLM。

# Installation
```shell
pip install vllm

# 如果考虑量化，例如 AWQ
pip install autoawq

# GPTQ量化方案
pip install optimum
pip install auto-gptq 
```

# Example
这里演示使用Qwen2.5-32B-AWQ量化模型进行推理，运行示例代码自动下载模型，如果设备GPU不存在，可以先通过命令行下下来。
```shell
# simple_example() 展示了vllm推理单条数据.
# simple_batch_chat() 展示了vllm 批次推理
python vllm_example.py
```
[option] 单独使用命令行下载模型：
```shell
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ --local-dir ./cache
# or GPTQ量化版本
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --local-dir ./cache
```

v100-32G 不支持AWQ量化, Volta 架构不支持，需要Ampere 架构.

LLM常用参数:

| 参数                     | 类型                  | 默认值           | 说明                                                                                                                                                                                                                                                                                                                              |
|:-----------------------|:--------------------|:--------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model                  | str or Path         | 必填	           | 模型的路径或名称。可以是本地路径，也可以是模型库中的模型名称（如 "Qwen/Qwen2.5-32B-Instruct-AWQ"）                                                                                                                                                                                                                                                               |
| tokenizer              | PreTrainedTokenizer | option        | Tokenizer 对象，用于将文本转换为模型输入格式。通常与模型一起加载。                                                                                                                                                                                                                                                                                          |
| quantization           | str                 | option        | 量化方法。可以是 "GPTQ"、"AWQ" 等，指定量化方案。                                                                                                                                                                                                                                                                                                 |
| max_model_len	         |                     |               | 模型可以推理的上下文长度，如果不填，将自动从模型的config文件中读取, 这个值与KvCache有关，且成正比例，如果一直OOM，可以通过降低max_model_len来降低KvCache的占用.                                                                                                                                                                                                                             |
| gpu_memory_utilization | float               | 0.9           | 提高gpu_memory_utilization的值会增加模型的 KV cache 大小。<br>KV cache 是模型推理中存储键值对的内存缓存，它帮助加速序列生成，因为它避免了重复计算。更大的 KV cache 可以减少重复计算，从而提高模型的吞吐量和推理速度。<br>如果设置较高的 gpu_memory_utilization，并且 GPU 内存足够，模型的推理效率会提高。<br>风险：如果 gpu_memory_utilization 设置得过高，尤其是在 GPU 内存不足的情况下，可能会导致 out-of-memory (OOM) 错误。这意味着模型会尝试在 GPU 上加载更多数据或缓存，而超出可用内存，导致计算失败。 |
| dtype  | str                 | torch.float16 | 使用"auto"将会自动从config文件中读取，如果config中是float32, 则会被使用float16替代。(真的很奇怪的设计，明明说了支持float16)                                                                                                                                                                                                                                             |
| max_seq_len_to_capture| int                 | option        | 假设 max_seq_len_to_capture=512。如果输入的序列长度小于或等于 512 tokens，CUDA 图将用于优化推理过程。如果输入的序列长度大于 512 tokens，则将退回到 eager 模式，性能可能下降。                                                                                                                                                                                                           |
|                        |                     |               |                                                                                                                                                                                                                                                                                                                                 |

Sampling常用参数:

| 参数                            | 类型          | 默认值   | 说明                                                         |
|:------------------------------|:------------|:------|:-----------------------------------------------------------|
| n                             | int         | 1     | 一条指令生成n个回复（适合用于强化训练采样）                                     |
| best_of                       | int or None | None  | 一条指令生成best_of个回复，但是只会返回n个最好的回复                             |
| 3 types of penalty            |             |       | 即vllm会通过设置的惩罚参数，按照设置的三种惩罚逻辑之一，对生成的token进行概率降低，使其后续再生成的概率变低 |
| temperature                   | float       |       | 随处可见                                                       |
| top_p                         | float       |       | 取值(0,1) 取这个概率范围内的所有tokens作为候选                              |
| top_k                         | int         |       | 取前top_k个tokens作为候选                                         |
| seed                          |             |       | 随机种子，方便复现                                                  |
| stop                          | list        | None  | 返回的输出不会有这个list中的值                                          |
| stop_token_ids                | list        | None  | 生成过程中如果产生了list中的值会立即停止输出，且返回值包含了这个停止值，除非是特殊的token才不会被包含    |
| bad_words                     | list        | None  | 不允许list中的词作为最后一个被生成的token                                  |
| include_stop_str_in_output    | bool        | False | 是否生成的结果包含停止词                                               |
| ignore_eos                    | bool        | False | 是否忽略EOS token，并且在生成EOS后继续生成token                           |
| max_tokens                    | int or None | 16    | 单个结果最大tokens长度                                             |
| min_tokens                    | int or None | 0     | 在EOS或停止词前可生成的最小长度                                          |
| detokenize                    | bool        | True  | 是否decode结果                                                 |
| skip_special_tokens           | bool        | True  | 是否跳过输出中的特殊token                                            |
| spaces_between_special_tokens | bool        | True  | 是否在连续的特殊token之间增加空格                                        |
| truncate_prompt_tokens        | int or None | None  | 整数则从后向前保留整数k个token，可以考虑每次使用len(prompt)                     |


# Refs
[1] [VLLM docs](https://docs.vllm.ai/en/latest/models/generative_models.html)<br>
[2] [QWEN docs](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)<br>
[3] [QWEN2.5-32B-AWQ HuggingFace Space](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)<br>
[4] [VLLM generate parameters docs](https://docs.vllm.ai/en/latest/usage/engine_args.html#engine-args)<br>
[5] [VLLM SamplingParams docs](https://docs.vllm.ai/en/latest/dev/sampling_params.html#vllm.SamplingParams)<br>
[6] [OpenAI API docs](https://platform.openai.com/docs/api-reference/realtime-server-events/response/text/done)<br>
