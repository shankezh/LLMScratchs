from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from utilities import check_network_params
from model.hf_gpt_model import GMQConfig, GMQModel
from model.hf_lmq_model import LMQModel,LMQConfig

def get_prompts():
    prompt_datas = [
        '特朗普',
        '美国',
        '中国',
        '英国',
        '人工智能是'
    ]
    return prompt_datas

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ######################################################
    # For GMQ model
    # model_file = "/root/autodl-tmp/gpt2/results_truncate/checkpoint-178829"
    # model_file = "./results_truncate/gmq_pretrain_truncate"
    # m_cfg, m_model,m_type = GMQConfig, GMQModel, GMQConfig.model_type

    ######################################################
    # For LMQ model
    model_file = "./LMQ-0.5B/lmq_pretrained"
    m_cfg, m_model, m_type = LMQConfig, LMQModel, LMQConfig.model_type
    ########################################################
    # 1. register customized config and model in huggingface
    AutoConfig.register(m_type, m_cfg)
    AutoModel.register(m_cfg, m_model)

    ##########################################################
    # For LMQ model, because model is saved by deepspeed, hence need transfer dtype
    config = AutoConfig.from_pretrained(model_file)
    if hasattr(config, "dtype") and isinstance(config.dtype, str):
        config.dtype = getattr(torch, config.dtype, torch.float32)

    ############################################################
    # 2. using QWEN2.5 Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ##############################################################
    # 3. Load model, set device and switch evaluation mode
    model = AutoModel.from_pretrained(model_file, config=config)
    model.to(device)
    model.eval()

    ###############################################################
    # 4. get total parameters info
    check_network_params(model)

    ################################################################
    # 5. initial test prompt and run test code
    eval_prompts = get_prompts()

    with torch.no_grad():
        for prompt in eval_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt",return_attention_mask=True).to(device)

            output = model.generate(
                input_ids=input_ids['input_ids'],
                max_length = 100,
                temperature = 1,
                attention_mask = input_ids['attention_mask'],
                top_k = 50,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample = True
            )
            decode_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(decode_text)