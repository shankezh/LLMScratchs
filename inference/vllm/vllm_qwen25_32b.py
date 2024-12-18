from transformers import AutoTokenizer,AutoModelForCausalLM
from vllm import LLM, SamplingParams

def init_model_and_tokenizer():
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./cache")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = init_model_and_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    llm = LLM(model = model, tokenizer = tokenizer)

    outputs = llm.generate("hello, how are you?")

    for output in outputs:
        prompt = output.prompt
        text = output.outputs[0].txt
        print(f"Prompt: {prompt!r}, Text: {text!r}")


