import deepspeed
import torch
import json
from torch.utils.data import DataLoader
from torch.distributed import get_rank
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
import shutil
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_lmq_model import LMQModel, LMQConfig
from utilities import IterReadDataset


# setting SFT mode: 0 - using general system template
#                   1 - using more specific system template
SFT_TRAINING_MODE = 0


def system_template(mode: int, type=None):
    # The fist SFT with using general template
    if mode == 0:
        system = """你是小辣，一名友好的AI助手！请根据用户的指令回答相关问题。"""
    else:
        # After first SFT using specific template
        task_dicts = {
            "NLI": "content"
        }
        system = task_dicts[type]
    return system


def fill_template(system, conversations):
    # system = system_template(SFT_TRAINING_MODE, system)
    if system == "simple_qa":
        system = system_template(SFT_TRAINING_MODE, system)
    template = f"<|im_start|>system\n{system}<|im_end|>"

    for message in conversations:
        role = message["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        # role = "user" if role == "human" else role
        content = message["value"]
        template += f"\n<|im_start|>{role}\n{content}<|im_end|>"
    template += "<|endoftext|>"

    return template


def tokenize_function(batch, tokenizer):
    # start_time = time.time()
    systems = [sample["system"] for sample in batch]
    conversations = [sample["conversations"] for sample in batch]

    templates = [
        fill_template(system, conversation) for system, conversation in zip(systems, conversations)
    ]


    # 使用每个批次的 max_length 进行填充
    tokens_data = tokenizer(
        templates,
        truncation=True,
        max_length=2048,
        padding=False,
        return_special_tokens_mask=False,
        return_attention_mask=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    tokenized_data = data_collator(tokens_data)
    return tokenized_data

def cus_collate_fn(tokenizer):
    def batch_collate_fn(batch):
        return tokenize_function(batch, tokenizer)
    return batch_collate_fn

def init_model_and_tokenizer(model_path):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_path is not None:
        m_cfg, m_model, m_type = LMQConfig, LMQModel, LMQConfig.model_type
        AutoConfig.register(m_type, m_cfg)
        AutoModel.register(m_cfg, m_model)
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "dtype") and isinstance(config.dtype, str):
            config.dtype = getattr(torch, config.dtype, torch.float32)
        model = AutoModel.from_pretrained(model_path, config=config)
        model = torch.compile(model)
    else:
        raise ValueError("Please provide model_path")
    return model, tokenizer


#############################################
# deepspeed do not have dtype class, so transfer it to string
#############################################
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
    # 将 epoch 和 step 作为检查点的标签
    tag = f"epoch_{epoch}_step_{step}"
    print(f"Saving checkpoint for epoch {epoch}, step {step}...")

    # 调用 DeepSpeed 的 save_checkpoint 方法
    model_engine.save_checkpoint(save_dir, tag)

    print(f"Checkpoint saved at {save_dir}, tag: {tag}")

    if get_rank() == 0:
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
        return json_data["train_micro_batch_size_per_gpu"], json_data["gradient_accumulation_steps"], json_data[
            "total_data_size"], json_data["val_data_size"]


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
    # 0. setting related files path and initial parameters
    ##########################################################
    # model_path = "../sft/LMQ-0.5B/lmq_pretrained"
    model_path = "Qwen/Qwen2.5-0.5B"
    train_data_path = "../data/sft_train_data.jsonl"
    val_data_path = "../data/sft_val_data.jsonl"
    ds_config_path = "ds_config.json"
    save_checkpoints_dir = "./checkpoints"
    save_final_model_dir = "./results/lmq_sft"
    max_checkpoints = 3  # 3 checkpoints maximum remained
    save_interval = 5000  # how many steps to save checkpoint once
    epoch_num = 1
    val_interval = 5000  # validate model once each 5000 steps
    val_after_step = 0  # validate only trigger after a certain step

    #############################################################
    # 1. calculate MAX_STEPS for Streaming datasets
    #############################################################
    gpu_num_devices = torch.cuda.device_count()
    print(f"current have {gpu_num_devices} GPU devices")

    train_micro_batch_size_per_gpu, gradient_accumulation_steps, total_data_size, val_data_size = get_json_param(
        ds_config_path)
    effective_batch_size = train_micro_batch_size_per_gpu * gpu_num_devices

    max_steps = total_data_size // effective_batch_size
    print(f"MAX_STEPS is {max_steps} ...")

    set_json_param_max_step(ds_config_path, max_steps)

    val_after_step = max_steps // 2  # validate event after half max_steps

    ###########################################################
    # 2. deepspeed need initial function for distributed()
    ###########################################################
    deepspeed.init_distributed()

    ###########################################################
    # 3. setting distributed environment
    # local_rank will get GPU items from "CUDA_VISIBLE_DEVICES"
    # eg: four GPUs, 0,1,2,3
    # shell >> CUDA_VISIBLE_DEVICES=1,2 deepspeed --num_gpus=2 deepspeed_pretrain.py
    # means local_rank also will show 0 and 1 items, because it will get infos from CUDA_VISIBLE_DEVICES
    ###########################################################
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    ###########################################################
    # 4. get model and tokenizer
    ###########################################################
    model, tokenizer = init_model_and_tokenizer(model_path)

    ###########################################################
    # 5. setting data and config tokenizer function
    ###########################################################
    # train_data_path = f"../data/sft_train_data{rank}.jsonl"
    # print(f"Rank[{rank}]: load {train_data_path}")
    train_dataset = IterReadDataset(file_path=train_data_path, total_lines=total_data_size, world_size=world_size, rank=rank)
    # It's small, do need spilt to three parts
    val_dataset = IterReadDataset(file_path=val_data_path,total_lines=total_data_size, world_size=1, rank=1)


    #############################################################
    # 6. warp datasets by DataLoader
    #############################################################
    collate_fn = cus_collate_fn(tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_micro_batch_size_per_gpu,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=train_micro_batch_size_per_gpu,
                                collate_fn=collate_fn, pin_memory=True)

    #############################################################
    # 7. construct DeepSpeed env
    #############################################################
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )

    #############################################################
    # 8. get distributed environment infos
    #    load checkpoints if existed else last_checkpoint is None
    #############################################################
    last_checkpoint = load_checkpoint(model_engine, save_checkpoints_dir)

    ##############################################################
    # 9. recovery epoch and step infos
    ##############################################################
    if last_checkpoint:
        # epoch, step = map(int, last_checkpoint.split('_')[1:][::2])    # 从检查点标签解析 epoch 和 step
        start_epoch, start_step = map(int, [last_checkpoint.split('_')[1], last_checkpoint.split('_')[3]])
        print(f"Resuming training from epoch {start_epoch}, step {start_step}")
    else:
        start_epoch, start_step = 0, 0

    #################################################################
    # 10. training logic
    ###################################################################
    for epoch in range(start_epoch, epoch_num):
        # make sure could be shuffled each epoch
        tokenized_train_dataset.set_epoch(epoch)
        model_engine.train()
        val_loss_ave = 0  # define a slot for saving validate loss
        for step, train_batch in enumerate(train_dataloader, start=0):
            # if step <= start_step:
            #     print(f"jump step {step}")
            #     continue


            # print(train_batch)
            # assign data to same device
            # for key, value in train_batch.items():
            #     print(f"Rank[{rank}]-{step}: {key}: shape={value.shape}, dtype={value.dtype}")


            train_batch = {k: v.to(model_engine.local_rank) for k, v in train_batch.items()}

            # forward propagate
            train_loss, _ = model_engine(input_ids=train_batch["input_ids"],
                                         attention_mask=train_batch["attention_mask"], labels=train_batch["input_ids"])

            # back propagate
            model_engine.backward(train_loss)

            # update gradient each gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                model_engine.step()

            # running validate after val_after_step
            if step > val_after_step and (step + 1) % val_interval == 0:
                print("start evaluated ...")
                val_loss_ave = 0
                model_engine.eval()
                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_dataloader):
                        val_batch = {k: v.to(model_engine.local_rank) for k, v in val_batch.items()}
                        val_loss, _ = model_engine(input_ids=val_batch["input_ids"],
                                                   attention_mask=val_batch["attention_mask"],
                                                   labels=val_batch["input_ids"])
                        print(
                            f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1}, val_step: {val_step + 1}, batch_loss: {val_loss.item():.4f}")
                        val_loss_ave += val_loss.item()
                val_loss_ave = val_loss_ave / val_data_size
                # switch to training mode
                model_engine.train()

            print(
                f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1} / {max_steps}, train_loss: {train_loss:.4f}, val_loss: {val_loss_ave:.4f}")

            if step != 0 and step % save_interval == 0 and rank==0:
                save_checkpoint_with_epoch(model_engine, save_checkpoints_dir, epoch, step, max_checkpoints)
                print("Save checkpoint ...")
    if rank == 0:
        save_model(model_engine, save_final_model_dir)