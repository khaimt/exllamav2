
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
app = FastAPI()
import datetime

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
batch_size = 9
config = ExLlamaV2Config()
config.model_dir = model_directory
config.max_batch_size = batch_size
config.no_flash_attn = True # this is a must based on an issue on github
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, batch_size=batch_size, lazy = True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
generator.warmup()

def get_prompt(query):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {query} ASSISTANT:"


test_cases = [
    "describe Hanoi and Vietnam in one sentence",
    "hi, how are you",
    "what is your favourite food?",
    "do you have a name?",
    "say hello in Vietnamese",
    "say hello in Vietnamese",
    "say hello in Vietnamese",
    "say hello in Vietnamese",
    "say hello in Vietnamese"
]
prompts = [get_prompt(case) for case in test_cases]
times = []
for i in range(1):
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.0000001
    settings.token_repetition_penalty = 1.0
    t1 = datetime.datetime.now()
    outputs = generator.generate_simple(prompts[: batch_size], settings, 512, seed = 1234, encode_special_tokens=True)
    t2 = datetime.datetime.now()
    times.append((t2 - t1).total_seconds())
    print("exe time: ", (t2 - t1).total_seconds())
avg_time = sum(times) / len(times)
print(f"avg time: {avg_time}, per_item: {avg_time/batch_size}")

for i in range(len(prompts[: batch_size])):
    output = outputs[i]
    prompt = prompts[i]
    print("+ ", output[len(prompt): ])
