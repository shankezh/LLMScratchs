import torch, json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
import deepspeed
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_lmq_model import LMQModel,LMQConfig

def tokenize_function(examples, tokenizer ):
    # 在每个文本的结尾添加 <|im_end|> 作为上下文结束标记
    texts_with_end_token = [text + "<|im_end|>" for text in examples["text"]]

    tokenized_outputs = tokenizer(
        texts_with_end_token,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    lengths = [len(tokens) for tokens in tokenized_outputs['input_ids']]
    max_length = min(max(lengths), 1024)


    tokens_data = tokenizer(
        texts_with_end_token,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=False,
        return_attention_mask=True)

    # 把 <|im_end|>（eos_token） 换成 <|endoftext|>（对应 pad_token）在预训练阶段
    # 遍历每个序列，替换其中的 eos_token_id 为 pad_token_id
    tokens_data['input_ids'] = [
        [tokenizer.pad_token_id if token_id == tokenizer.eos_token_id else token_id for token_id in seq]
        for seq in tokens_data['input_ids']
    ]
    return tokens_data


def prepare_data(data_path, tokenizer):
    train_dataset = load_dataset("json", keep_in_memory=True, data_files=data_path, split="train",
                                 streaming=True)
    # 使用批处理进行标记化，填充
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"])
    return tokenized_train_dataset

def collate_fn(batch):
    # batch是一个list，其中每个元素是从数据集返回的字典，如:
    # {"input_ids": [...], "attention_mask": [...]}
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def init_model_and_tokenizer(model_path = None):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_path is not None:
        # 注册自定义配置类
        AutoConfig.register("llama3_2_mix_qwen2_5", LMQConfig)
        # 注册自定义模型类
        AutoModel.register(LMQConfig, LMQModel)
        model = AutoModel.from_pretrained(model_path)
    else:
        model_config = LMQConfig(
            vocab_size=151665,
            emb_dim=1024,
            n_heads=16,
            n_layers=18,
            hidden_dim=4096,
            context_length=2048,
            n_kv_groups=8,
            rope_base=5000,
            rope_freq={
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_context_length": 2048,
            },
            dtype=torch.float32
        )
        model = LMQModel(config=model_config)

    torch.compile(model)
    return model, tokenizer


def save_model(model_engine, url):
    # 确保配置中的 dtype 是字符串格式
    if hasattr(model_engine.module.config, "dtype"):
        model_engine.module.config.dtype = str(model_engine.module.config.dtype)
    model_engine.module.save_pretrained("./results/lmq_pretrained")

def save_checkpoint_with_epoch(model_engine, save_dir, epoch, step, max_checkpoints=3):
    """
    保存 DeepSpeed 模型检查点，并将 epoch 和 step 信息包含在文件名中，限制最大保存数量。
    
    参数:
        model_engine: DeepSpeedEngine 对象，用于保存检查点。
        save_dir (str): 检查点保存的目录。
        epoch (int): 当前 epoch。
        step (int): 当前 step。
        max_checkpoints (int): 最大检查点保存数量，超过时删除旧的检查点。
    """
    import os
    import shutil
    # 将 epoch 和 step 作为检查点的标签
    tag = f"epoch_{epoch}_step_{step}"
    print(f"Saving checkpoint for epoch {epoch}, step {step}...")

    # 调用 DeepSpeed 的 save_checkpoint 方法
    model_engine.save_checkpoint(save_dir, tag)

    print(f"Checkpoint saved at {save_dir}, tag: {tag}")

    # 获取当前保存的所有检查点目录
    checkpoints = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
    
    # 检查点按创建时间排序（旧的在前）
    checkpoints.sort(key=lambda d: os.path.getctime(os.path.join(save_dir, d)))

    # 如果保存的检查点数量超过 max_checkpoints，则删除最早的检查点
    while len(checkpoints) > max_checkpoints:
        oldest_checkpoint = checkpoints.pop(0)
        oldest_path = os.path.join(save_dir, oldest_checkpoint)
        print(f"Removing old checkpoint: {oldest_path}")
        shutil.rmtree(oldest_path)

def load_checkpoint(model_engine, save_dir):
    try:
        checkpoints = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
        if not checkpoints:
            print("No checkpoints found to load.")
            return None

        latest_checkpoint = max(checkpoints, key=lambda d: os.path.getctime(os.path.join(save_dir, d)))
        print(f"Loading checkpoint: {latest_checkpoint}")
        model_engine.load_checkpoint(save_dir, latest_checkpoint)
    except:
        print("No checkpoints found to load or broken.")
        return None
    return latest_checkpoint

def get_json_param(url):
    with open(url, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data["train_micro_batch_size_per_gpu"], json_data["gradient_accumulation_steps"], json_data["total_data_size"]

def set_json_param_max_step(url, max_steps):
    with open(url, 'r', encoding='utf-8') as fr:
        config = json.load(fr)
    config["scheduler"]["params"]["total_num_steps"] = max_steps
    with open(url, 'w', encoding='utf-8') as fw:
        json.dump(config, fw, indent=2)
        print(f"Successfully saved updated config to {url}.")

if __name__ == '__main__':
    ###########################################################
    # concrete random seed, make sure process can be repeated#
    ###########################################################
    torch.manual_seed(123)

    ###########################################################
    # 1. deepspeed need initial function for distributed()
    ###########################################################
    deepspeed.init_distributed()

    ###########################################################
    # 2. setting distributed environment
    # local_rank will get GPU items from "CUDA_VISIBLE_DEVICES"
    # eg: four GPUs, 0,1,2,3
    # shell >> CUDA_VISIBLE_DEVICES=1,2 deepspeed --num_gpus=2 deepspeed_pretrain.py
    # means local_rank also will show 0 and 1 items, because it will get infos from CUDA_VISIBLE_DEVICES
    ###########################################################
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    ###########################################################
    # 3. get model and tokenizer
    ###########################################################
    model_path = None
    model, tokenizer = init_model_and_tokenizer(model_path)

    ###########################################################
    # 4. setting data and config tokenizer function
    ###########################################################
    data_path = "../data/pretrain_train.json"
    tokenized_train_dataset = prepare_data(data_path, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


    save_dir = "./checkpoints"
    ds_config_path = "ds_config.json"
    max_checkpoints = 3  # 最多保留 3 个检查点
    save_interval = 5000  # 每隔 $ 个 steps 保存一次
    epoch_num = 1
    
    gpu_num_devices = torch.cuda.device_count() # GPU数量
    print(f"current have {gpu_num_devices} GPU devices")
    
    train_micro_batch_size_per_gpu, gradient_accumulation_steps, total_data_size = get_json_param(ds_config_path)

    effective_batch_size = train_micro_batch_size_per_gpu  * gpu_num_devices
    # 计算最大步数
    max_steps = total_data_size // effective_batch_size
    print(f"Max stpes is {max_steps} ... ")

    set_json_param_max_step(ds_config_path, max_steps)
    #
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_micro_batch_size_per_gpu, collate_fn=data_collator)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # optimizer=optimizer,
        config = ds_config_path
    )

    # 获取分布式环境信息
    rank = torch.distributed.get_rank()
    last_checkpoint = load_checkpoint(model_engine, save_dir)

    if last_checkpoint:
        epoch, step = map(int, last_checkpoint.split('_')[1:][::2])
    else:
        epoch, step = 0, 0

    for epoch in range(epoch_num):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            # DeepSpeed会自动分配设备，无需手动 to(device)
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            # 前向传播
            loss, _ = model_engine(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["input_ids"])
            # 反向传播与优化
            model_engine.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0:
                model_engine.step()

            print(f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1} / {max_steps}, loss {loss.item():.4f}")

            # 每隔 save_interval 步保存一次检查点
            if step % save_interval == 0:
                save_checkpoint_with_epoch(model_engine, save_dir, epoch, step, max_checkpoints)


    # 使用DeepSpeed保存模型
    # DeepSpeed支持使用 model_engine.save_checkpoint 来保存权重
    # 或使用 model_engine.module.save_pretrained 来使用transformers的save_pretrained。
    # 这里假设模型是transformers格式，可以直接调用：
    if rank == 0:
        # model_engine.save_checkpoint("./results/")
        save_model(model_engine, "./results/lmq_pretrained")