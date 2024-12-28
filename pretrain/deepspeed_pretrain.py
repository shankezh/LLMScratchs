import torch, json
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
import deepspeed
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_lmq_model import LMQModel,LMQConfig
from utilities import IterReadDataset

def tokenize_function(batch, tokenizer ):
    texts_with_end_token = [sample["text"]  for sample in batch]

    tokens_data = tokenizer(
        texts_with_end_token,
        truncation=True,
        max_length=1024,
        padding=False,
        return_special_tokens_mask=False,
        return_attention_mask=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    tokenized_data = data_collator(tokens_data)
    return tokenized_data



def cus_collate_fn(tokenizer):
    def batch_collate_fn(batch):
        return tokenize_function(batch, tokenizer)
    return batch_collate_fn

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

    model = torch.compile(model)
    return model, tokenizer


def save_model(model_engine, url):
    # 确保配置中的 dtype 是字符串格式
    if hasattr(model_engine.module.config, "dtype"):
        model_engine.module.config.dtype = str(model_engine.module.config.dtype)
    model_engine.module.save_pretrained(url)

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
    # 0. setting distributed environment
    # local_rank will get GPU items from "CUDA_VISIBLE_DEVICES"
    # eg: four GPUs, 0,1,2,3
    # shell >> CUDA_VISIBLE_DEVICES=1,2 deepspeed --num_gpus=2 deepspeed_pretrain.py
    # means local_rank also will show 0 and 1 items, because it will get infos from CUDA_VISIBLE_DEVICES
    ###########################################################
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    ###########################################################
    # 1. deepspeed need initial function for distributed()
    ###########################################################
    deepspeed.init_distributed()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print(f"[Init] rank={rank}, local_rank={local_rank}, cuda current device={torch.cuda.current_device()}")

    ###########################################################
    # 2. setting related files path and initial parameters
    ##########################################################
    model_save_path = "./results/lmq_pretrained"
    data_path = "../data/pretrain_train.json"
    save_dir = "./checkpoints"
    ds_config_path = "ds_config.json"
    max_checkpoints = 3  # 最多保留 3 个检查点
    save_interval = 5000  # 每隔 $ 个 steps 保存一次
    epoch_num = 1

    ###########################################################
    # 3. calculate MAX_STEPS for streaming datasets
    ###########################################################
    gpu_num_devices = torch.cuda.device_count()  # GPU数量
    print(f"current have {gpu_num_devices} GPU devices")

    train_micro_batch_size_per_gpu, gradient_accumulation_steps, total_data_size = get_json_param(ds_config_path)
    effective_batch_size = train_micro_batch_size_per_gpu * gpu_num_devices
    max_steps = total_data_size // effective_batch_size
    print(f"Max stpes is {max_steps} ... ")

    set_json_param_max_step(ds_config_path, max_steps)

    ###########################################################
    # 4. get model and tokenizer
    ###########################################################
    model_path = None
    model, tokenizer = init_model_and_tokenizer(model_path)

    ###########################################################
    # 5. setting data and config tokenizer function
    ###########################################################
    train_datasets = IterReadDataset(file_path=data_path, total_lines=total_data_size, world_size=world_size, rank=rank)
    collate_fn = cus_collate_fn(tokenizer)
    train_dataloader = DataLoader(train_datasets, batch_size=train_micro_batch_size_per_gpu, collate_fn=collate_fn)


    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # optimizer=optimizer,
        config = ds_config_path
    )


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
                model_engine.zero_grad()

            print(f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1} / {max_steps}, loss {loss.item():.4f}")

            # 每隔 save_interval 步保存一次检查点
            if step % save_interval == 0 and step !=0:
                save_checkpoint_with_epoch(model_engine, save_dir, epoch, step, max_checkpoints)


    if rank == 0:
        # model_engine.save_checkpoint("./results/")
        save_model(model_engine, model_save_path)