import torch, time
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from model.hf_gpt_model import GMQModel,GMQConfig


default_system = "你的名字是良智,是一个擅长回答问题的AI助手,请一步步地思考然后再帮助用户回答问题."
pre_sys = "你叫良智，是一名多功能的 AI 助手，当前的任务类型是："

def fill_template_(user, assistant):
    system = pre_sys + "实体识别。"
    template = f"<|im_start|>system\n{system}<|im_end|>"
    template += f"\n<|im_start|>user\n{user}<|im_end|>"
    template += f"\n<|im_start|>assistant\n{assistant}<|im_end|><|endoftext|>"
    return template


def get_data_from_json(path):
    train_data = load_dataset("json", keep_in_memory=True, data_files=path)
    chosen_list, rejected_list = [], []
    for idx in range(train_data.num_rows['train']):
        prompt = train_data['train']['conversations'][idx][0]['value']
        chosen = train_data['train']['chosen'][idx]['value']
        rejected = train_data['train']['rejected'][idx]['value']
        chosen_msg = fill_template_(prompt, chosen)
        rejected_msg = fill_template_(prompt, rejected)
        chosen_list.append(chosen_msg)
        rejected_list.append(rejected_msg)
    return chosen_list, rejected_list

def get_tokens_from_data(tokenizer, chosen_list, rejected_list):
    tokenized_chosen_pre = tokenizer(
        chosen_list,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    tokenized_rejected_pre = tokenizer(
        rejected_list,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    lengths_chosen = [len(tokens) for tokens in tokenized_chosen_pre['input_ids']]
    lengths_rejected = [len(tokens) for tokens in tokenized_rejected_pre['input_ids']]

    max_length_chosen = min(max(lengths_chosen), 1024)
    max_length_rejected = min(max(lengths_rejected), 1024)

    tokenized_chosen_data = tokenizer(
        chosen_list,
        truncation=True,
        max_length=max_length_chosen,
        padding='max_length',
        return_attention_mask=True,
        return_special_tokens_mask=False
    )

    tokenized_rejected_data = tokenizer(
        rejected_list,
        truncation=True,
        max_length=max_length_rejected,
        padding='max_length',
        return_attention_mask=True,
        return_special_tokens_mask=False
    )

    return tokenized_chosen_data, tokenized_rejected_data


def get_models_and_tokenizer(device):

    # 注册自定义配置类
    AutoConfig.register("gpt_mix_qwen", GMQConfig)
    # 注册自定义模型类
    AutoModel.register(GMQConfig, GMQModel)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_file = "../sft/results_sft/gmq_sft_scene_NER"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tr_model = GMQModel.from_pretrained(model_file)
    tr_model.to(device)
    ref_model = GMQModel.from_pretrained(model_file)
    ref_model.to(device)
    return tr_model, ref_model, tokenizer

if __name__ == '__main__':
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    po_model, ref_model, tokenizer = get_models_and_tokenizer(device)


    chosen_data, rejected_data = get_data_from_json("../data/dpo_data.json")

    optimizer = AdamW(po_model.parameters(), lr=)

