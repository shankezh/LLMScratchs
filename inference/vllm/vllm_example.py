from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def init_llm_and_tokenizer(model_path: str, model_name, quantization: str, max_model_len: int, dtype: str):
    llm = LLM(model=model_path, quantization=quantization, max_model_len=max_model_len, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache")
    return llm, tokenizer


def build_gpt_format(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    return messages


def build_sampling_params(temperature=0.7, top_p=0.8, max_tokens=512):
    return SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def simple_batch_chat():
    #######################################################
    # Qwen2.5 series tokenizer are same
    # using local model
    model_path = "./cache"
    model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    llm, tokenizer = init_llm_and_tokenizer(model_path=model_path, model_name=model_name, quantization="GPTQ",
                                            max_model_len=2048, dtype="auto")
    sampling_params = build_sampling_params(temperature=0.7, top_p=0.8, max_tokens=2048)

    zero_shot_prompt = r'你是zero-shot COT任务翻译大师，当前需要做数据处理，需要对链式思考任务进行翻译，要求翻译链式思考相关输入任务中的英语，将其转换成中文，要求翻译合理，通顺，流程且意思不变，请注意，你只需要翻译input和output对应的部分不需要进行额外扩展或改写，务必使用中文。结果依旧使用"input-cn": "翻译内容" 和 "output-cn": "翻译内容" 进行输出.'

    input = "请一步一步认真地思考下面的问题：\nFran is in charge of counting votes for the book club's next book, but she always cheats so her favorite gets picked. Originally, there were 10 votes for Game of Thrones, 12 votes for Twilight, and 20 votes for The Art of the Deal. Fran throws away 80% of the votes for The Art of the Deal and half the votes for Twilight. What percentage of the altered votes were for Game of Thrones?"
    ans = "First find the total number of The Art of the Deal votes Fran throws away: 80% * 20 votes = 16 votes. Then subtract these votes from the total number of The Art of the Deal votes to find the altered number: 20 votes - 16 votes = 4 votes. Then divide the total number Twilight votes by 2 to find the altered number of votes: 12 votes / 2 = 6 votes. Then add the altered number of votes for each book to find the total altered number of votes: 6 votes + 4 votes + 10 votes = 20 votes. Then divide the number of votes for Game of Thrones by the total altered number of votes and multiply by 100% to express the answer as a percentage: 10 votes / 20 votes * 100% = 50%.\n因此，最终的答案是：\n50"

    order = f"input:\n{input}\noutput:{ans}"

    message = build_gpt_format(system=zero_shot_prompt, user=order)

    batch_message = [message for _ in range(5)]

    outputs = llm.chat(messages=batch_message, sampling_params=sampling_params, use_tqdm=True)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text}")


def simple_example():
    #######################################################
    # Qwen2.5 series tokenizer are same
    # using local model
    model_path = "./cache"
    model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    llm, tokenizer = init_llm_and_tokenizer(model_path=model_path, model_name=model_name, quantization="GPTQ",
                                            max_model_len=4096, dtype="auto")

    sampling_params = build_sampling_params(temperature=0.7, top_p=0.8, max_tokens=512)

    zero_shot_prompt = r'你是zero-shot COT任务翻译大师，当前需要做数据处理，需要对链式思考任务进行翻译，要求翻译链式思考相关输入任务中的英语，将其转换成中文，要求翻译合理，通顺，流程且意思不变，请注意，你只需要翻译input和output对应的部分不需要进行额外扩展或改写，务必使用中文。结果依旧使用"input-cn": "翻译内容" 和 "output-cn": "翻译内容" 进行输出.'

    input = "请一步一步认真地思考下面的问题：\nFran is in charge of counting votes for the book club's next book, but she always cheats so her favorite gets picked. Originally, there were 10 votes for Game of Thrones, 12 votes for Twilight, and 20 votes for The Art of the Deal. Fran throws away 80% of the votes for The Art of the Deal and half the votes for Twilight. What percentage of the altered votes were for Game of Thrones?"
    ans = "First find the total number of The Art of the Deal votes Fran throws away: 80% * 20 votes = 16 votes. Then subtract these votes from the total number of The Art of the Deal votes to find the altered number: 20 votes - 16 votes = 4 votes. Then divide the total number Twilight votes by 2 to find the altered number of votes: 12 votes / 2 = 6 votes. Then add the altered number of votes for each book to find the total altered number of votes: 6 votes + 4 votes + 10 votes = 20 votes. Then divide the number of votes for Game of Thrones by the total altered number of votes and multiply by 100% to express the answer as a percentage: 10 votes / 20 votes * 100% = 50%.\n因此，最终的答案是：\n50"

    order = f"input:\n{input}\noutput:{ans}"

    message = build_gpt_format(system=zero_shot_prompt, user=order)

    batch_message = []
    tokenized_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    batch_message.append(tokenized_text)

    outputs = llm.generate(batch_message, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    simple_batch_chat()