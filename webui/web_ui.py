from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.hf_lmq_model import LMQModel,LMQConfig
import chainlit

def init_model_and_tokenizer(model_path):
    m_cfg, m_model, m_type = LMQConfig, LMQModel, LMQConfig.model_type
    AutoConfig.register(m_type, m_cfg)
    AutoModel.register(m_cfg, m_model)
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "dtype") and isinstance(config.dtype, str):
        config.dtype = getattr(torch, config.dtype, torch.float32)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModel.from_pretrained(model_path, config=config)
    model.to(device)
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

######################################################
# For LMQ model
# model_path = "../sft/results/lma_sft"
model_path = "/mnt/d/Code/LLM/LLMScratchs/sft/results/lma_sft"
model, tokenizer = init_model_and_tokenizer(model_path)
########################################################



def process_message(prompt):
    system = "你是“小辣”，一个友好的智能助手!"
    template = f"<|im_start|>system\n{system}<|im_end|>"
    template += f"\n<|im_start|>user\n{prompt}<|im_end|>"
    template += f"\n<|im_start|>assistant\n"
    input_len = len(template)
    input_ids = tokenizer(template, return_tensors="pt", return_attention_mask=True)
    print("start work ....")

    output = model.generate(
        input_ids=input_ids['input_ids'],
        max_length=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        attention_mask=input_ids['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    decode_text = tokenizer.decode(output[0], skip_special_tokens=False)
    # print(decode_text)
    return decode_text[input_len:]

def test_message(prompt):
    return prompt

@chainlit.on_message
async def main(message: chainlit.Message):
    decode_text = await chainlit.make_async(process_message)(message.content)
    print(decode_text)
    await chainlit.Message(
        content = f"{decode_text}"
    ).send()
