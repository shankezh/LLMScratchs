import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
import deepspeed
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_lmq_model import LMQModel,LMQConfig
from utilities import IterReadDataset
from ds_untilities import save_checkpoint_with_epoch, load_checkpoint, get_json_param, set_json_param_max_step, save_model

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
    train_datasets = IterReadDataset(file_path=data_path, total_lines=total_data_size, world_size=world_size, rank=rank, offset_seed=0,buffer_size=0)
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
                # model_engine.zero_grad()

            print(f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1} / {max_steps}, loss {loss.item():.4f}")

            # 每隔 save_interval 步保存一次检查点
            if step % save_interval == 0 and step !=0:
                save_checkpoint_with_epoch(model_engine, save_dir, epoch, step, max_checkpoints)


    if rank == 0:
        # model_engine.save_checkpoint("./results/")
        save_model(model_engine, model_save_path)