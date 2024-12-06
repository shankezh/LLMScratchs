import torch
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer, AutoConfig,AutoModel
from datasets import load_dataset
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_gpt_model import GMQModel,GMQConfig



# 当前预训练数据单组最大长度是708个token，+1个endoftext
max_tokens_length_in_data =  708 + 1
# train_size = 4, 828, 394

def init_model_and_tokenizer(device, model_path = None):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_path is not None:
        # 注册自定义配置类
        AutoConfig.register("gpt_mix_qwen", GMQConfig)
        # 注册自定义模型类
        AutoModel.register(GMQConfig, GMQModel)
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
    else:
        model_config = GMQConfig(
            vocab_size=len(tokenizer), # 不要使用tokenizer.vocab_size,这样就不包含了特殊标记
            emb_dim=360,
            n_layers=12,
            n_heads=12,
            context_length=1024,
            drop_rate=0.0,
            qkv_bias=False,
        )
        model = GMQModel(config=model_config).to(device)
    return model, tokenizer

def get_training_args():
    # total_data_size = 5364883   #预训练数据总条数
    total_data_size =4828394
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 8

    # 计算总步数
    num_devices = 1
    # 计算每步的 token 数量
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_devices

    # 计算最大步数
    max_steps = total_data_size // effective_batch_size
    # max_steps = (total_data_size // tokens_per_step) * num_epochs


    training_args = TrainingArguments(
        output_dir="./results_truncate",  # 输出模型和检查点的目录
        overwrite_output_dir=True,
        # num_train_epochs=num_epochs,
        max_steps=max_steps,  # 使用 max_steps 替代 num_train_epochs
        per_device_train_batch_size=per_device_train_batch_size,  # 可以根据你的 GPU 调整这个值
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="./logs",  # 日志保存目录
        logging_steps=64,
        save_steps=640,
        save_total_limit=3,
        learning_rate=1e-3,
        weight_decay=0.01,
        torch_compile=True,
        max_grad_norm=1.0,  # 启用梯度裁剪，限制最大梯度范数为1.0
        fp16=True,  # 是否使用混合精度
        # lr_scheduler_type="constant",  # 禁用学习率衰减，保持学习率固定
        # resume_from_checkpoint="last-checkpoint", # 直接指定恢复的检查点

        # 新增的参数
        eval_strategy="steps",  # 评估策略：'no'、'steps'、'epoch'
        eval_steps=3200,  # 每隔多少步进行一次评估
        eval_accumulation_steps=gradient_accumulation_steps,

        # load_best_model_at_end=True,  # 在训练结束后加载最佳模型
        # metric_for_best_model="loss",  # 以验证集损失作为评估指标
        # greater_is_better=False  # 对于损失，越低越好
    )
    return training_args

def tokenize_function(examples, tokenizer ):
    # 在每个文本的结尾添加 <|im_end|> 作为上下文结束标记
    texts_with_end_token = [text + tokenizer.eos_token for text in examples["text"]]
    tokens_data = tokenizer(texts_with_end_token, truncation=True, max_length=max_tokens_length_in_data, padding="max_length",
              return_special_tokens_mask=False, return_attention_mask=True)
    # 把 <|im_end|>（eos_token） 换成 <|endoftext|>（对应 pad_token）在预训练阶段
    # 遍历每个序列，替换其中的 eos_token_id 为 pad_token_id
    tokens_data['input_ids'] = [
        [tokenizer.pad_token_id if token_id == tokenizer.eos_token_id else token_id for token_id in seq]
        for seq in tokens_data['input_ids']
    ]
    return tokens_data


def data_test(flag):
    if flag:
        for i, example in enumerate(train_dataset):
            print(example)
            if i == 1:
                break
        print("Tokenized Dataset 样本:")
        sidx = 2
        for i, example in enumerate(tokenized_train_dataset):
            if i < sidx:
                print(example['input_ids'])
                continue
            print(len(example['input_ids']))
            print(example)
            if i == sidx:
                break

if __name__ == '__main__':
    # torch.manual_seed(123)

    data_test_flag = False
    # model_path = "./results/gmq_pretrain"
    model_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = init_model_and_tokenizer(device,model_path)
    training_args = get_training_args()

    # 训练数据
    train_dataset = load_dataset("json", keep_in_memory=True, data_files="../data/pretrain_train.json", split="train", streaming=True)

    # 验证数据
    eval_dataset = load_dataset("json", keep_in_memory=True,data_files = "../data/pretrain_val.json", split="train", streaming=True)

    # 使用批处理进行标记化，填充
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"])

    # 仅采样百分之一数量作为评估
    # sample_size = (5364883 - 4828394)//100
    sample_size = 128
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    ).take(sample_size)


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # False表示是因果语言模型建模（GPT风格预训练）
    )
    if data_test_flag:
        data_test(True)
    else:
        trainer = Trainer(
            model = model,
            args = training_args,
            data_collator = data_collator,
            train_dataset = tokenized_train_dataset,
            eval_dataset = tokenized_eval_dataset,
        )

        # trainer.train(resume_from_checkpoint=True)
        trainer.train()
        trainer.save_model("./results_truncate/gmq_pretrain_truncate")
