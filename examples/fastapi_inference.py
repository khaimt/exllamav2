import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
app = FastAPI()

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
model_directory = os.environ["MODEL_PATH"]
print("Loading model: " + model_directory)
config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
generator.warmup()

t2 = datetime.datetime.now()
print("loading time: ", (t2 - t1).total_seconds())

class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.001
    max_new_token: int = 512


@app.post("/generate")
async def generate(prequest: PromptRequest):
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = prequest.temperature
    settings.token_repetition_penalty = 1.0
    #settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    output = generator.generate_simple(prequest.prompt, settings, prequest.max_new_token, seed = 1234)
    return {"result": output[len(prequest.prompt): ]}
