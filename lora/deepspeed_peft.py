from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
import deepspeed
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from utilities import IterReadDataset
from ds_untilities import save_checkpoint_with_epoch, load_checkpoint, get_json_param, set_json_param_max_step, save_model

def init_peft_config():
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r = 8,
        lora_alpha= 32,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    return peft_config

# device is None for multi-gpus training
def init_model_and_tokenizer(model_path, peft_config,device=None):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",cache_dir="../pretrain/cache/QWEN2.5-0.5B", trust_remote_code=True)
        print(type(model.parameters()))
        model = get_peft_model(model, peft_config)
        print(type(model.parameters()))
        # print(f"Model Infos: {model.print_model_infos()}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {trainable_params}")
    else:
        raise ValueError("Please provide model_path")
    return model, tokenizer


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
    # model_path = "../sft/LMQ-0.5B/lmq_pretrained"
    model_path = "Qwen/Qwen2.5-0.5B"
    train_data_path = "../data/sft_train_data.jsonl"
    val_data_path = "../data/sft_val_data.jsonl"
    ds_config_path = "ds_config_qwen2_5.json"
    save_checkpoints_dir = "./checkpoints"
    save_final_model_dir = "./results/qwen25_0p5B"
    max_checkpoints = 3  # 3 checkpoints maximum remained
    save_interval = 5000  # how many steps to save checkpoint once
    epoch_num = 1
    val_interval = 5000  # validate model once each 5000 steps
    val_after_step = 0  # validate only trigger after a certain step

    #############################################################
    # 3. calculate MAX_STEPS for Streaming datasets
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
    # 4. get model and tokenizer
    ###########################################################
    peft_config = init_peft_config()
    model, tokenizer = init_model_and_tokenizer(model_path, peft_config)

    ###########################################################
    # 5. setting data and config tokenizer function
    ###########################################################
    # train_data_path = f"../data/sft_train_data{rank}.jsonl"
    # print(f"Rank[{rank}]: load {train_data_path}")
    train_dataset = IterReadDataset(file_path=train_data_path, total_lines=total_data_size, world_size=world_size,
                                    rank=rank, offset_seed=0, buffer_size=0)
    # It's small, do need spilt to three parts
    val_dataset = IterReadDataset(file_path=val_data_path, total_lines=val_data_size, world_size=1, rank=0)

    #############################################################
    # 6. warp datasets by DataLoader
    #############################################################
    collate_fn = cus_collate_fn(tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_micro_batch_size_per_gpu,
                                  collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_micro_batch_size_per_gpu,
                                collate_fn=collate_fn, pin_memory=True)

    #############################################################
    # 7. construct DeepSpeed env
    #############################################################
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
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
        start_epoch, start_step = 0, -1

    #################################################################
    # 10. training logic
    ###################################################################
    for epoch in range(start_epoch, epoch_num):

        model_engine.train()
        val_loss_ave = 0  # define a slot for saving validate loss
        for step, train_batch in enumerate(train_dataloader, start=0):
            if step <= start_step:
                print(f"jump step {step}")
                continue


            # print(train_batch)
            # assign data to same device
            # for key, value in train_batch.items():
            #     print(f"Rank[{rank}]-{step}: {key}: shape={value.shape}, dtype={value.dtype}")


            train_batch = {k: v.to(model_engine.local_rank) for k, v in train_batch.items()}

            # forward propagate
            output = model_engine(input_ids=train_batch["input_ids"],
                                         attention_mask=train_batch["attention_mask"], labels=train_batch["input_ids"])
            train_loss = output.loss

            # back propagate
            model_engine.backward(train_loss)

            # update gradient each gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                model_engine.step()
                # model_engine.zero_grad()

            # running validate after val_after_step
            if step > val_after_step and (step + 1) % val_interval == 0:
                print("start evaluated ...")
                val_loss_ave = 0
                model_engine.eval()
                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_dataloader):
                        val_batch = {k: v.to(model_engine.local_rank) for k, v in val_batch.items()}
                        val_output = model_engine(input_ids=val_batch["input_ids"],
                                                   attention_mask=val_batch["attention_mask"],
                                                   labels=val_batch["input_ids"])
                        print(
                            f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1}, val_step: {val_step + 1}, batch_loss: {val_output.loss.item():.4f}")
                        val_loss_ave += val_output.loss.item()
                val_loss_ave = val_loss_ave / val_data_size
                # switch to training mode
                model_engine.train()

            print(
                f"Rank[{rank}]: Epoch {epoch + 1}, step {step + 1} / {max_steps}, train_loss: {train_loss:.4f}, val_loss: {val_loss_ave:.4f}")

            if step != 0 and step % save_interval == 0:
                save_checkpoint_with_epoch(model_engine, save_checkpoints_dir, epoch, step, max_checkpoints)
                print("Save checkpoint ...")
    if rank == 0:
        save_model(model_engine, save_final_model_dir)

