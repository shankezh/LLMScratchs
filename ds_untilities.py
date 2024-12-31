import torch
from torch.distributed import get_rank
import shutil, os, json

####################################################################################
# All functions for DeepSpeed
####################################################################################



####################################################################################
# save Deepspeed checkpoint, the epoch and step infos will record with file name.
# and only allow number of "max_checkpoints" existed, the latest checkpoint would
# replace (cover) oldest checkpoint
###################################################################################
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
    torch.distributed.barrier() # 同步等待所有GPU进程
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
    torch.distributed.barrier()  # 同步等待所有GPU进程

######################################################################
# load checkpoint, it would automatically check checkpoints files,
# and use the latest checkpoint to recovery training.
######################################################################
def load_checkpoint(model_engine, save_dir):
    try:
        rank = get_rank()
        latest_checkpoint = None

        if rank == 0:
            if not os.path.exists(save_dir):
                print(f"Rank[{rank}]: Checkpoint Directory does not exist! Making directory")
            else:
                checkpoints = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
                latest_checkpoint = max(checkpoints, key=lambda d: os.path.getmtime(os.path.join(save_dir, d)))
                if latest_checkpoint == save_dir:
                    latest_checkpoint = None
        obj = [latest_checkpoint]
        torch.distributed.broadcast_object_list(obj, src=0)
        latest_checkpoint = obj[0]

        if latest_checkpoint is None:
            print(f"Rank[{rank}]: No valid checkpoints found after broadcast.")
            return None
        torch.distributed.barrier()
        print(f"Rank[{rank}]:Loading checkpoint: {latest_checkpoint}")
        model_engine.load_checkpoint(save_dir, latest_checkpoint)
        torch.distributed.barrier()
    except Exception as e:
        print(f"Failed to load checkpoint due to error: {e}")
        return None
    return latest_checkpoint


#############################################
# deepspeed do not have dtype class, so transfer it to string
#############################################
def save_model(model_engine, url):
    # 确保配置中的 dtype 是字符串格式
    if hasattr(model_engine.module.config, "dtype"):
        model_engine.module.config.dtype = str(model_engine.module.config.dtype)
    model_engine.module.save_pretrained(url)


###########################################################################
# To read JSON (ds_config.json)
############################################################################
def get_json_param(url):
    with open(url, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data["train_micro_batch_size_per_gpu"], json_data["gradient_accumulation_steps"], json_data[
            "total_data_size"], json_data["val_data_size"]

###########################################################################
# To write MAX_STEPS in JSON (ds_config.json)
############################################################################
def set_json_param_max_step(url, max_steps):
    with open(url, 'r', encoding='utf-8') as fr:
        config = json.load(fr)
    config["scheduler"]["params"]["total_num_steps"] = max_steps
    with open(url, 'w', encoding='utf-8') as fw:
        json.dump(config, fw, indent=2)
        print(f"Successfully saved updated config to {url}.")