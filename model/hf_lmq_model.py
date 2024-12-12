from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer,GenerationMixin
import torch.nn as nn
import torch
from transformers.modeling_outputs import CausalLMOutput
from llama3_2_model import TransformerBlock, RMSNorm

from LLM.LLMScratchs.utilities import check_network_params

class LMQConfig(PretrainedConfig):
    model_type = 'llama3_2_mix_qwen2_5'

    def __init__(self,
                 vocab_size=50257,  # 默认值，用户可覆盖
                 emb_dim=768,
                 n_heads=12,
                 n_layers=12,
                 hidden_dim=3072,
                 context_length=512,
                 n_kv_groups=4,
                 rope_base=10000,
                 rope_freq=None,
                 dtype=torch.float32,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.n_kv_groups = n_kv_groups
        self.rope_base = rope_base
        self.rope_freq = rope_freq or {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_context_length": 512}
        self.dtype = dtype

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

class LMQModel(PreTrainedModel):
    config_class = LMQConfig
    base_model_prefix = "transformer"  # 这是 Hugging Face 的标准字段，便于正确识别模型

    def __init__(self, config):
        super().__init__(config)
        self.emb_tok = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.emb_dim, dtype=config.dtype)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias = False, dtype=config.dtype)

    def forward(self, input_ids, attention_mask=None, labels = None, return_dict=False):
        x = self.emb_tok(input_ids).to(dtype=self.layers[0].norm1.weight.dtype)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        # 以下改动是为了兼容huggingface的训练器接口
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 使用 attention_mask 来掩盖 pad 部分的损失
            if attention_mask is not None:
                # print("计算一次mask")
                # 将 attention_mask 拉平并扩展到所有词汇
                attention_mask = attention_mask[..., 1:].contiguous().view(-1)  # shift_attention_mask
                loss = loss * attention_mask  # 只保留有效部分的损失，填充部分的损失为 0

                # 对所有有效部分求平均损失, 即重新归一化loss损失
                loss = loss.sum() / attention_mask.sum()

        # 如果要求返回字典，则使用 CausalLMOutput 结构返回
        if return_dict:
            return CausalLMOutput(loss=loss, logits=logits) if labels is not None else CausalLMOutput(logits=logits)
        return (loss, logits) if loss is not None else logits

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _reorder_cache(self, past, beam_idx):
        # 如果你支持 beam search，你可以实现这个方法
        return past

if __name__ == '__main__':
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_cfg = LMQConfig(
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

    model = LMQModel(model_cfg).to(device)
    check_network_params(model)
    model.eval()

    start_context = "你好，请说你好！"

    input_ids = tokenizer(start_context, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids=input_ids,
        max_length=30,
        temperature=1,
        top_k=50,
        eos_token_id=tokenizer.pad_token_id,  # 预训练是pad_token_id <|endoftext|>，sft对应的是eos_token_id <|im_end|>
        do_sample=True
    )

    generated_tokens = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_tokens)
