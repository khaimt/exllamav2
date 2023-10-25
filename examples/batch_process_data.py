
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
app = FastAPI()
import datetime
import json 
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

import datetime
from pydantic import BaseModel


# Initialize model and cache
t1 = datetime.datetime.now()
model_directory = "/workspace/exllamav2/WizardLM-70B-V1.0-GPTQ"
print("Loading model: " + model_directory)
batch_size = 8
config = ExLlamaV2Config()
config.model_dir = model_directory
config.max_batch_size = batch_size
config.no_flash_attn = True
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, batch_size=batch_size, lazy = True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
generator.warmup()


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def save_json(data, path):
    with open(path, "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))
        

def get_qa_prompt(question, context):
    result = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: You are an intelligent assistant that can generate the answer to a question based only on the provided knowledge. If you don't know the answer or cannot extract the answer from the provided knowledge, just say that you don't know, don't try to make up an answer. Note that the answer must be based on the provided knowledge.
=======
Here is the knowledge:
{context}
=======
Here is the question: 
{question}
=======
Now please generate the answer to this question as an assistant following this format:
+ Thought: extracting the relevant information to the question from the provided knowledge and then generate the reasoning to answer the question
+ Answer: based on Thought, provide the complete answer to the question; if the provided knowledge doesn't contain the answer or cannot reason to get the answer, please say that you cannot answer this based on your knowledge
-------------- ASSISTANT:+ Thought:"""
    return result


def get_batches(mini_size, total):
    iter_num = total // mini_size
    result = []
    for i in range(iter_num + 1):
        start = i * mini_size
        end = i * mini_size + mini_size
        if end > total:
            end = total
        if end > start:
            result.append((start, end))
    return result


def get_prompt_from_item(item, item_index):
    result = []
    for q_index, sub in enumerate(item["sub_questions"]):
        question = sub["question"]
        paragraph = sub["paragraph"]
        prompt = get_qa_prompt(question, paragraph)
        result.append({"prompt": prompt, "item_index": item_index, "q_index": q_index})
    return result


def process_prompts(prompts):
    l_prompts = prompts
    size = len(prompts)
    if size < batch_size:
        l_prompts = prompts + [prompts[-1] for _ in range(batch_size - size)]  # add to have batch_size
    assert len(l_prompts) == batch_size
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.0000001
    settings.token_repetition_penalty = 1.0
    outputs = generator.generate_simple(l_prompts, settings, 1024, seed = 1234, encode_special_tokens=True)
    assert len(outputs) == len(l_prompts) == batch_size
    result = []
    for prompt, output in zip(l_prompts, outputs):
        result.append(output[len(prompt): ])
    return result[: size]


def parse_llm_output(output):
    items = output.split("+ Answer:")
    if len(items) == 2:
        return items[0].strip(), items[1].strip()
    print("-----------cannot parse llm_output: ---------")
    print(output)
    return None, None
    


def infer_data(input_path, current_path, save_path):
    input_items = read_json(input_path)
    output_items = read_json(current_path)
    p_items = input_items[len(output_items): ]
    print("number of items to process: ", len(p_items))
    total_prompts = []
    for index, item in enumerate(p_items):
        ps = get_prompt_from_item(item, index)
        total_prompts.extend(ps)
    print("number of prompts: ", len(total_prompts))
    batches = get_batches(batch_size, len(total_prompts))
    count = 0
    t1 = datetime.datetime.now()
    for start, end in batches:
        prompts = total_prompts[start: end]
        p_texts = [p["prompt"] for p in prompts]
        outputs = process_prompts(p_texts)
        assert len(outputs) == len(p_texts) == len(prompts)
        for i in range(len(outputs)):
            thought, answer = parse_llm_output(outputs[i])
            item_index = prompts[i]["item_index"]
            q_index = prompts[i]["q_index"]
            if thought is not None:
                final_answer = thought + "\nAnswer: " + answer
                p_items[item_index]["sub_questions"][q_index]["long_answer"] = final_answer
                p_items[item_index]["meta_info"]["llm"] = "direct"
            
        save_json(output_items + p_items[: item_index + 1], save_path)
        count += 1
        t2 = datetime.datetime.now()
        total_time = (t2 - t1).total_seconds()
        avg_time = total_time / count
        remain_count = len(batches) - count
        print(f"{count}/{len(batches)}, avg_time = {avg_time}, remaining time: {remain_count * avg_time}, item_index: {item_index} / {len(p_items)}")            


def main():
    infer_data("../data/raw_multi_hop_qa.json", "../data/final.json", "../data/result.json")


if __name__ == "__main__":
    main()
