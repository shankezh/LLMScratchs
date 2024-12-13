import torch, json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding
from datasets import load_dataset
import argparse
import sys,os

from LLM.business.dpo.ttt import train_data

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


def prepare_data(data_path):
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


def init_model_and_tokenizer(device, model_path = None):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_path is not None:
        # 注册自定义配置类
        AutoConfig.register("llama3_2_mix_qwen2_5", LMQConfig)
        # 注册自定义模型类
        AutoModel.register(LMQConfig, LMQModel)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
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
        model = LMQModel(config=model_config).to(device)

    torch.compile(model)
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='For DeepSpeed Pretrain')
    # 每个训练进程在所有参与训练的进程中会有一个全局唯一的序号，通常从0开始递增至world_size - 1（world_size是所有参与训练的进程总数）。
    # global rank用于在全局范围内区分各个进程。例如，当有多机多卡训练时，global rank 可以区分不同机器上的进程。

    # local rank 通常从0开始，对应该机器上的第 0 个 GPU，第 1 个 GPU，以此类推。这样，每个进程就可以通过 local rank 来知道自己应该使用哪块 GPU
    parser.add_argument("--local_rank", type=int, default=0, help = "本地进程号")

    args = parser.parse_args()


    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = None
    model, tokenizer = init_model_and_tokenizer(model_path)
    data_path = ""
    tokenized_train_dataset = prepare_data(data_path)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    batch_size = 6
    #
    # train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size, collate_fn=data_collator)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # epoch_num = 1
    # max_steps = 5364883 // batch_size
    # for epoch in range(epoch_num):
    #     for step, batch in enumerate(train_dataloader):
    cmd_args = {
        "train_batch_size": 6,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4,
            }
        },
        "fp16":{
            "enabled": True
        },
        "zero_optimization": True
    }

    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args = cmd_args,
        model=model,
        model_parameters=model.parameters(),
        train_data = tokenized_train_dataset,
        config = "ds_config.json"
    )


    epoch_num = 1
    # 假设 max_steps 计算方式与原来相同
    max_steps = 5364883 // batch_size

    for epoch in range(epoch_num):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            if step >= max_steps:
                break

            # DeepSpeed会自动分配设备，无需手动 to(device)
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            # 前向传播
            loss, _ = model_engine(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["input_ids"])
            # 反向传播与优化
            model_engine.backward(loss)
            model_engine.step()

            print(f"Epoch {epoch + 1}, step {step + 1} / {max_steps}, loss {loss.item():.4f}")

    # 使用DeepSpeed保存模型
    # DeepSpeed支持使用 model_engine.save_checkpoint 来保存权重
    # 或使用 model_engine.module.save_pretrained 来使用transformers的save_pretrained。
    # 这里假设模型是transformers格式，可以直接调用：
    model_engine.module.save_pretrained("./results/lmq_pretrained")