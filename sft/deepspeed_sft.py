import deepspeed
import torch
import os
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
from model.hf_lmq_model import LMQModel, LMQConfig

# setting SFT mode: 0 - using general system template
#                   1 - using more specific system template
SFT_TRAINING_MODE = 0

def system_template(mode:int, type=None):
    # The fist SFT with using general template
    if mode == 0:
        system = """你是“小辣”，一个友好的智能助手，擅长处理以下任务：自然语言推理、文本摘要、对联生成、音乐评论、实体识别、关键词识别、文本纠错、情感分析、文案生成、链式思考、开放问答、古诗仿写、文本相似度匹配、歌词生成、阅读理解、文言文翻译、作文生成、金庸风格续写、自我介绍。请根据用户的输入内容，理解任务并提供相应的回答。
        """
    else:
        # After first SFT using specific template
        task_dicts = {
            "NLI":"content"
        }
        system = task_dicts[type]
    return system

def fill_template(system, conversations):

    system = system_template(SFT_TRAINING_MODE, system)
    template = f"<|im_start|>system\n{system}{tokenizer.eos_token}"

    for message in conversations:
        role = message["from"]
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        # role = "user" if role == "human" else role
        content = message["value"]
        template += f"\n<|im_start|>{role}\n{content}{tokenizer.eos_token}"
        template += tokenizer.pad_token
    return template

def tokenize_function(examples, tokenizer):
    # start_time = time.time()
    systems = examples["system"]
    conversations = examples["conversations"]

    templates = [
        fill_template(system, conversation) for system, conversation in zip(systems, conversations)
    ]

    return tokenizer(
        templates,
        truncation=True,
        max_length=2048,
        padding="False",
    )



def prepare_data(train_data_path, val_data_path, tokenizer):
    train_dataset = load_dataset("json", keep_in_memory=True, datafile_path=train_data_path, split="train", streaming=True)
    val_dataset = load_dataset("json", keep_in_memory=True, datafile_path=val_data_path, split="train", streaming=True)


    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    tokenized_val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    return tokenized_train_dataset, tokenized_val_dataset


def init_model_and_tokenizer(model_path):
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
        raise Exception("Please provide model_path")
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
    train_data_path = "../data/pretrain_train.json"
    val_data_path = "../data/pretrain_val.json"
    ds_config_path = "ds_config.json"
    save_checkpoints_dir = "./checkpoints"
    save_final_model_dir = "./results/lma_sft"
    max_checkpoints = 3   # 3 checkpoints maximum remained
    save_interval = 5000  # how many steps to save checkpoint once
    epoch_num = 1
    val_interval = 5000     # validate model once each 5000 steps
    val_after_step = 0      # validate only trigger after a certain step



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
    tokenized_train_dataset, tokenized_val_dataset = prepare_data(train_data_path, val_data_path, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    #############################################################
    # 5. calculate MAX_STEPS for Streaming datasets
    #############################################################
    gpu_num_devices = torch.cuda.device_count()
    print(f"current have {gpu_num_devices} GPU devices")

    train_micro_batch_size_per_gpu, gradient_accumulation_steps, total_data_size, val_data_size = get_json_param(ds_config_path)
    effective_batch_size = train_micro_batch_size_per_gpu * gpu_num_devices

    max_steps = total_data_size // effective_batch_size
    print(f"MAX_STEPS is {max_steps} ...")

    val_after_step = max_steps // 2  # validate event after half max_steps

    #############################################################
    # 6. warp datasets by DataLoader
    #############################################################
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_micro_batch_size_per_gpu, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_val_dataset, batch_size=train_micro_batch_size_per_gpu, collate_fn=data_collator)

    #############################################################
    # 7. construct DeepSpeed env
    #############################################################
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters = model.parameters(),
        config = ds_config_path
    )

    #############################################################
    # 8. get distributed environment infos
    #    load checkpoints if existed else last_checkpoint is None
    #############################################################
    rank = torch.distributed.get_rank()
    last_checkpoint = load_checkpoint(model_engine, save_final_model_dir)


    ##############################################################
    # 9. recovery epoch and step infos
    ##############################################################
    if last_checkpoint:
        epoch, step = map(int, last_checkpoint.split('_')[1:][::2])
    else:
        epoch, step = 0, 0

    #################################################################
    # 10. training logic
    ###################################################################
    for epoch in range(epoch_num):
        model_engine.train()
        val_loss_ave = 0    # define a slot for saving validate loss
        for step, train_batch in enumerate(train_dataloader):

            # assign data to same device
            train_batch = {k:v.to(model_engine.local_rank) for k, v in train_batch.items()}

            # forward propagate
            train_loss, _ = model_engine(input_ids=train_batch["input_ids"], attention_mask=train_batch["attention_mask"], labels=train_batch["input_ids"])

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
                for val_step, val_batch in enumerate(val_dataloader):
                    val_batch = {k:v.to(model_engine.local_rank) for k, v in val_batch.items()}
                    val_loss, _ = model_engine(input_ids=val_batch["input_ids"], attention_mask=val_batch["attention_mask"], labels=val_batch["input_ids"])
                    print(f"Rank[{rank}]: Epoch {epoch+1}, step {step+1}, val_step: {val_step+1}, batch_loss: {val_loss.item():.4f}")
                    val_loss_ave += val_loss.item()
                val_loss_ave = val_loss_ave / val_data_size
                # switch to training mode
                model_engine.train()

            print(f"Rank[{rank}]: Epoch {epoch+1}, step {step+1} / {max_steps}, train_loss: {train_loss:.4f}, val_loss: {val_loss_ave:.4f}")

            if step % save_interval == 0:
                save_checkpoint_with_epoch(model_engine, save_checkpoints_dir, epoch, step,max_checkpoints)
                print("Save checkpoint ...")
    if rank == 0:
        save_model(model_engine, save_final_model_dir)