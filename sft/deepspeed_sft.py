import deepspeed
import torch
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from model.hf_lmq_model import LMQModel, LMQConfig


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
