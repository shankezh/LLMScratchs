import torch, time
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_gpt_model import GMQModel,GMQConfig



default_system = "你的名字是良智,是一个擅长回答问题的AI助手,请一步步地思考然后再帮助用户回答问题."
pre_sys = "你叫良智，是一名多功能的 AI 助手，当前的任务类型是："

task_descriptions = {
    "NLI": "自然语言推理。",
    "TextMatching": "文本匹配。",
    "StoryGeneration": "故事生成。",
    "ProductDesc": "商品文案生成。",
    "Summary": "文本摘要。",
    "AncientPoem": "古诗生成。",
    "NER": "实体识别。",
    "SentimentAnalyze": "情感分析。",
    "TextCorrection": "文本纠错。",
    "Couplet": "对对联。",
    "MusicComment": "生成音乐热评。",
    "KeywordRecognition": "关键词识别。",
    "ClassicalChinese": "翻译成文言文。",
    "Cot": "思维链思考。",
    "LyricGeneration": "歌词生成。",
    "Translation": "中英翻译。",
    "OpenQA": "开放问答。",
    "Composition": "作文生成。",
    "MRC": "阅读理解。",
    "JinYongGeneration": "金庸风格小说生成。",
    "Dictionary": "成语释义。",
}

def fill_template(system, conversations):
    if not system:
        system = default_system
    else:
        system = pre_sys + task_descriptions.get(system, default_system)
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
    # print(f"First few templates: {templates[:3]}")  # 调试输出模板
    # print(templates)
    # 获取每个样本的 token 长度，并限制最大长度为 1024
    tokenized_outputs = tokenizer(
        templates,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    # print(f"Tokenized outputs: {tokenized_outputs['input_ids'][:3]}")  # 调试输出 tokenized 信息

    # print(tokenized_outputs)
        # 计算每个样本的长度
    lengths = [len(tokens) for tokens in tokenized_outputs['input_ids']]
    max_length = min(max(lengths), 1024)
    # print(f"max_length is : {max_length} tokens in current batch.")
    # 使用每个批次的 max_length 进行填充
    tokens_data = tokenizer(
        templates,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=False,
        return_attention_mask=True
    )
    # print(f"Tokens data: {tokens_data['input_ids'][:3]}")  # 调试输出处理后的数据

        # 记录结束时间
    # end_time = time.time()

    # 打印时间差
    # processing_time = end_time - start_time
    # print(f"Time taken to process one batch: {processing_time:.4f} seconds")
    return tokens_data

def init_model_and_tokenizer(llm_path, tokenizer_path, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)
    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)
    model = AutoModel.from_pretrained(llm_path)
    model.to(device)
    return model, tokenizer


def get_training_args():
    # total_data_size = 7144091
    # total_data_size = 1649399
    total_data_size = 50000
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 8
    # 计算总步数
    num_devices = 1
    # 计算每步的 token 数量
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_devices
    # 计算最大步数
    max_steps = total_data_size // effective_batch_size

    config = TrainingArguments(
        output_dir="./results_sft",
        overwrite_output_dir=True,
        # num_train_epochs=num_epochs,
        max_steps=max_steps,  # 使用 max_steps 替代 num_train_epochs
        per_device_train_batch_size=per_device_train_batch_size,  # 可以根据你的 GPU 调整这个值
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="./logs",  # 日志保存目录
        logging_steps=64,
        save_steps=1280,
        save_total_limit=5,
        learning_rate=3e-6,
        weight_decay=0.01,
        torch_compile=True,
        max_grad_norm=1.0,  # 启用梯度裁剪，限制最大梯度范数为1.0
        fp16=True,  # 是否使用混合精度
        # lr_scheduler_type="constant",  # 禁用学习率衰减，保持学习率固定
        # resume_from_checkpoint="last-checkpoint", # 直接指定恢复的检查点

        # # 新增的参数
        # eval_strategy="steps",  # 评估策略：'no'、'steps'、'epoch'
        # eval_steps=3200,  # 每隔多少步进行一次评估
        # eval_accumulation_steps=gradient_accumulation_steps,
    )
    return config


if __name__ == '__main__':
    # torch.manual_seed(123)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # llm_model_path = "../pretrain/results/gmq_pretrain_truncate"
    llm_model_path = "./results_sft_temp/gmq_sft"
    tokenizer_model_path = "Qwen/Qwen2.5-0.5B-Instruct"

    model, tokenizer = init_model_and_tokenizer(llm_model_path, tokenizer_model_path, device)

    # 训练数据
    train_dataset = load_dataset("json", keep_in_memory=True, data_files="../data/sft_data_NER.json", split="train",
                                 streaming=True)

    # # 验证数据
    # eval_dataset = load_dataset("json", keep_in_memory=True, data_files="pretrain_data_lines.json", split="train",
    #                             streaming=True)
    # 使用批处理进行标记化，填充
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    # tokenized_eval_dataset = eval_dataset.map(
    #     lambda x: tokenize_function(x, tokenizer),
    #     batched=True,
    # )


        

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # False表示是因果语言模型建模（GPT风格预训练）
    )
    
    train_config = get_training_args()

    trainer = Trainer(
        model=model,
        args = train_config,
        data_collator = data_collator,
        train_dataset=tokenized_train_dataset,
        # eval_dataset=tokenized_eval_dataset,
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./results_sft/gmq_sft_scene_NER")


