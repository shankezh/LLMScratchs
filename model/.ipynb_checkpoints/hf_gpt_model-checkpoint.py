from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer,GenerationMixin
import torch.nn as nn
import torch
from transformers.modeling_outputs import CausalLMOutput
from model.gpt_model import TransformerBlock, LayerNorm


class GMQConfig(PretrainedConfig):
    model_type = 'gpt_mix_qwen'

    def __init__(self, vocab_size=50257, emb_dim=768, n_layers=12, n_heads=12, context_length=1024, drop_rate=0.1, qkv_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_length = context_length
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


class GMQModel(PreTrainedModel, GenerationMixin):
    config_class = GMQConfig
    base_model_prefix = "transformer"  # 这是 Hugging Face 的标准字段，便于正确识别模型

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.drop_rate)

        # 使用 ModuleList 而不是 Sequential 来允许在生成时使用。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config)
        # self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)  # 修改名称为 lm_head
        # 初始化权重
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels = None, return_dict=False):
        batch_size, seq_len = input_ids.shape
        tok_embeds = self.tok_emb(input_ids)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
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
    # here give the example
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "Qwen/Qwen2.5-0.5B"  # 这两个测试是一样的
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = GMQConfig(
        vocab_size=len(tokenizer), # 不要使用tokenizer.vocab_size,这样就不包含了特殊标记
        emb_dim=768,
        n_layers=12,
        n_heads=12,
        context_length=1024,
        drop_rate=0.0,
        qkv_bias=False,
    )
    # print(len(tokenizer))
    model = GMQModel(model_config).to(device)
    model.eval()

    start_context = "你好，请说你好！"

    input_ids = tokenizer(start_context, return_tensors="pt").input_ids.to(device)

    output = model.generate(
        input_ids = input_ids,
        max_length=30,
        temperature=1,
        top_k=50,
        eos_token_id = tokenizer.pad_token_id, # 预训练是pad_token_id <|endoftext|>，sft对应的是eos_token_id <|im_end|>
        do_sample=True
    )
    print(output)
    generated_tokens = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_tokens)



