# Introduction
这个文档用于描述DeepSpeed多卡训练的安装和使用过程

# installation
官方库的地址为:
[DeepSpeed Github](https://github.com/microsoft/DeepSpeed) <br>
页面有安装方法,pip和编译库文件方法自行选择。<br>
推荐使用docker安装，避免环境不一致问题：
```shell
pip install deepspeed
```
pycharm如何连接docker环境：
[Link ](https://www.cnblogs.com/lantingg/p/14927981.html)<br>


# Using
## 构建参数配置
DeepSpeed支持args和config两种形式的配置模式，二选1即可，这里使用config_json文件形式，参数选用见官网API [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)<br>

```json
{
  "train_batch_size": 144,
  "train_micro_batch_size_per_gpu": 24,
  "gradient_accumulation_steps": 6,
  "optimizer": {
    "type": "AdamW",
    "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-7,
        "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "total_num_steps": 37256,
      "warmup_min_ratio": 0.0,
      "warmup_num_steps": 1000,
      "cos_min_ratio": 0.0001,
      "warmup_type": "linear"
    }
  },
  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  },
  "steps_per_print": 10,
  "wall_clock_breakdown": false,
  "dump_state": false,
  "wandb": {
    "enabled": true,
    "team": "hogenzhu2023-university-of-sheffield",
    "project": "DS_Pre",
    "group": "2GPUs"
    }
}
```
Note: train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs.<br>
以下表格分别展示了单GPU，单机多GPU，多机多卡(2台机器，每台机器有4GPUs)的配置情况 <br>

| options | train_batch_size | train_micro_batch_size_per_gpu | gradient_accumulation_steps | commits                  |
|---------|------------------|--------------------------------|-----------------------------|--------------------------|
| 单机单卡    | 64               | 8                              | 8                           | 1 GPU                    |
| 单机多卡    | 64               | 8                              | 2                           | 4 GPUs                   |
| 多机多卡    | 64               | 8                              | 1                           | 2 Nodes with 4 GPUs/node |

Optimizer优化器可以直接使用torch的API在声明对象后传入deepspeed.initial()传入，也可以使用DeepSpeed自己实现的算子写在config配置中，DeepSpeed的算子API [DeepSpeed Optimizers](https://deepspeed.readthedocs.io/en/latest/optimizers.html) <br>
直接传入如下，这样config文件中就不需要写optimizer的配置了：
```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config = "ds_config.json"
    )
```
Schedule调度器配置可以按照DeepSpeed的支持文档进行选择和填写 [DeepSpeed Scheduler](https://deepspeed.readthedocs.io/en/latest/schedulers.html)<br>
FP16 training options 和 AMP 二选1即可, amp和ZeRO不能同时使用<br>
AMP优化级别 （[Nvidia API Link](https://nvidia.github.io/apex/amp.html#apex.amp.initialize)）:<br>

| opt_level         |意义|场景|优缺点|
|-------------------|---|---|---|
| "O0"              |完全使用 FP32（单精度）|默认配置，用于不需要混合精度的情况，例如测试或验证代码时|优点：数值稳定。缺点：无法享受 AMP 的性能提升 |
| "O1"              |混合精度：大多数计算使用 FP16，但关键运算（如损失计算、梯度累积）保留 FP32|推荐用于大部分任务，兼顾数值稳定性和性能提升|优点：数值稳定，适配范围广，训练速度显著提升。缺点：相比 O2 或 O3，性能提升略小。|
| "O2"              |混合精度：更激进地使用 FP16，只在必要时切换到 FP32，例如归一化层和特定精度敏感操作|对显存和性能要求更高的任务（如大模型训练），且对数值精度敏感的操作有限|优点：显存占用更小，性能提升更大。缺点：可能出现数值不稳定，需要手动调整梯度缩放等技巧|
| "O3"              |完全使用 FP16（纯半精度）|需要极致性能优化的场景，例如大批量分布式训练且能够容忍数值不稳定性的问题|优点：显存占用最低，性能最大化。缺点：数值不稳定，需要大量手动调试梯度缩放和代码改动|

gradient_clipping梯度剪裁，避免梯度爆炸，默认是1.0<br>
ZeRO内存优化，许多内容可以看API文档，里面有写到，这里使用一些可能是积极影响的参数：
> contiguous_gradients:true 拷贝梯度时候按会走，避免碎片影响，属于性能优化
> overlap_comm:true  通信和计算重叠可以在多卡训练时提高性能

Logging模块<br>
> steps_per_print: 表示训练N次打印一次训练信息<br>
> wall_clock_breakdown: 如果设置为 true，DeepSpeed 会启用前向传播、反向传播和优化器更新 各阶段的时间测量。输出的结果可以帮助用户了解各个阶段的耗时，从而定位性能瓶颈
> dump_state:如果设置为 true，DeepSpeed 会在初始化后打印出其内部状态信息，包括优化器、模型分布、ZeRO 设置等细节,用于检查 DeepSpeed 的初始化状态，确认所有配置（如 ZeRO stage、优化器参数、AMP 设置等）是否生效

## 启动训练
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 deepspeed_pretrain.py
# --nproc_per_node=2：表示在当前节点（机器）上启动两个进程，每个进程绑定到一块GPU。
# 手动指定 GPU，使用 CUDA_VISIBLE_DEVICES
```
或者:
```shell
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 deepspeed_pretrain.py
```
当然，你也可以只使用1个GPU进行训练
```shell
CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1 deepspeed_pretrain.py
```
运行成功后，可以看到：
```shell
Rank[0]: Epoch 1, step 7059 / 37256, loss 3.7812
Rank[1]: Epoch 1, step 7060 / 37256, loss 3.1094
...
# Rank [0/1]代表起的相关进程
```


# Refs
[1] [DeepSpeed Github](https://github.com/microsoft/DeepSpeed) <br>
[2] [other tutorial](https://github.com/OvJat/DeepSpeedTutorial)<br>
