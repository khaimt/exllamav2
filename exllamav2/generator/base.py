from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
import torch
import random

import torch.nn.functional as F

class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.tensor = None

    def __init__(self, model, cache, tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer


    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 2), dtype = torch.long)
        self.model.forward(input_ids, cache = None, input_mask = None, preprocess_only = True)


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len


    # TODO: Argument to allow different random samples over batch dimension

    def generate_simple(self, prompt: str or list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed = None,
                        token_healing = False,
                        encode_special_tokens = False,
                        decode_special_tokens = False,
                        loras = None):

        # Accept LoRA or list of LoRAs
        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids = self.tokenizer.encode(prompt, encode_special_tokens = encode_special_tokens)

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]

        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin gen

        self._gen_begin_base(ids, mask, loras)

        # Begin filters

        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        gen_settings.begin_filters(heal)

        # Generate tokens
        stops = [False] if isinstance(prompt, str) else [False for _ in range(len(prompt))]
        for i in range(num_tokens):

            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask = mask, loras = loras).float().cpu()
            token, _, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token = unhealed_token)
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)
            for j in range(len(stops)):
                if token[j].tolist()[0] == self.tokenizer.eos_token_id:
                    #print(f"{j} stop")
                    #print(token)
                    stops[j] = True
            if all(stops):
                #print("stop at: ", i)
                break
            gen_settings.feed_filters(token)

            unhealed_token = None
            if eos: break

        # Decode

        text = self.tokenizer.decode(self.sequence_ids, decode_special_tokens = decode_special_tokens)

        if isinstance(prompt, str): return text[0]
        return text


    def _gen_begin_base(self, input_ids, mask = None, loras = None):

        self.cache.current_seq_len = 0
        self.model.forward(input_ids[:, :-1], self.cache, input_mask = mask, preprocess_only = True, loras = loras)

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids

