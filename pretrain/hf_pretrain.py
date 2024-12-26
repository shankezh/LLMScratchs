import torch
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer, AutoConfig,AutoModel
from datasets import load_dataset
from itertools import chain
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig

def preprocess_and_concatenate(dataset, tokenizer, max_length=1024):
    """
    对数据进行预处理和拼接，将多个段落拼接到一起，并在段落之间插入 <|endoftext|>。
    """
    end_of_text_token = tokenizer.pad_token_id  # 获取 <|endoftext|> 的 ID

    def group_texts(examples):
        # 将所有文本拼接为一个长序列
        tokenized_inputs = tokenizer(examples["text"], add_special_tokens=False, truncation=False)["input_ids"]
        # 在拼接的每个段落之间插入 <|endoftext|>
        concatenated = list(chain(*[chunk + [end_of_text_token] for chunk in tokenized_inputs]))

        # 按 max_length 切分序列
        total_length = len(concatenated)
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        result = [
            concatenated[i : i + max_length]
            for i in range(0, total_length, max_length)
        ]
        return {"input_ids": result}

    return dataset.map(group_texts, batched=True, remove_columns=["text"])

def init_model_and_tokenizer(device, model_path = None):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_path is not None:
        AutoConfig.register("gpt_mix_qwen", GMQConfig)
        AutoModel.register(GMQConfig, GMQModel)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
    else:
        model_config = GMQConfig(
            vocab_size=len(tokenizer), # 不要使用tokenizer.vocab_size,这样就不包含了特殊标记
            emb_dim=512,
            n_layers=8,
            n_heads=8,
            context_length=1024,
            drop_rate=0.0,
            qkv_bias=False,
        )
        model = GMQModel(config=model_config).to(device)
    return model, tokenizer

def get_training_args():
    total_data_size = 5364883   #预训练数据总条数
    avg_token_per_sample = 195  # 打印前1000条数据处理得到的平均值
    max_length = 1024  # 最大序列长度
    per_device_train_batch_size = 10
    gradient_accumulation_steps = 5
    # 每条数据包含 <|endoftext|>，所以实际 token 数 = avg_token_per_sample + 1
    tokens_per_sample = avg_token_per_sample + 1
    # 计算总 token 数量
    total_tokens = total_data_size * tokens_per_sample

    effective_batch_size = per_device_train_batch_size
    tokens_per_step = effective_batch_size * max_length
    # To get max step based on total tokens number
    max_steps = total_tokens // tokens_per_step


    training_args = TrainingArguments(
        output_dir="./results",  # 输出模型和检查点的目录
        overwrite_output_dir=True,
        # num_train_epochs=num_epochs,
        max_steps=max_steps,  # 使用 max_steps 替代 num_train_epochs
        per_device_train_batch_size=per_device_train_batch_size,  # 可以根据你的 GPU 调整这个值
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="./logs",  # 日志保存目录
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        learning_rate=2e-7,
        weight_decay=0.01,
        torch_compile=True,
        max_grad_norm=1.0,  # 启用梯度裁剪，限制最大梯度范数为1.0
        fp16=True,  # 是否使用混合精度
        # lr_scheduler_type="constant",  # 禁用学习率衰减，保持学习率固定
        # resume_from_checkpoint="./results/gmq_pretrain", # 直接指定恢复的检查点
    )
    return training_args

def tokenize_function(examples, tokenizer ):
    return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length",
                     return_special_tokens_mask=True)



if __name__ == '__main__':
    # torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #########################################################
    # IMPORTANT: If you want to train model with multi-GPU, please check deepspeed_guide.md !!!
    #
    # 0. load model re-train or load_checkpoint to train
    #    also can choose None re-start
    ##########################################################
    # model_path = "./tempsaving/gmq_pretrain"
    # model_path = "./tempsaving/checkpoint-10500"
    model_path = None
    data_path = "../data/pretrain_train.json"
    model_save_path = "./results/gmq_pretrain"

    ###########################################
    # 1. init model and tokenizer
    ###########################################
    model, tokenizer = init_model_and_tokenizer(device, model_path)
    training_args = get_training_args()

    ###########################################
    # 2. data process with streaming form
    ###########################################
    streaming_dataset = load_dataset("json", data_files=data_path, split="train", streaming=True, keep_in_memory=True)

    ###########################################
    # To concatenate data make sure each data can be max_length
    ###########################################
    tokenized_dataset = preprocess_and_concatenate(streaming_dataset, tokenizer)

    ###########################################
    # wrap data base GPT form, actually it only pad the batch data
    ###########################################
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # False表示是因果语言模型建模（GPT风格预训练）
    )

    ##################################################################################
    # 3. initial trainer
    ######################################################################################
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        train_dataset = tokenized_dataset
    )

    ######################################################################################
    # 4. start training
    ######################################################################################
    trainer.train()

    ######################################################################################
    # 5. save model
    ######################################################################################
    trainer.save_model(model_save_path)
