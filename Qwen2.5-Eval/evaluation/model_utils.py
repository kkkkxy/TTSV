"""
https://github.com/allenai/open-instruct
"""
import torch
import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList
from pathlib import Path
import numpy as np
import torch.nn as nn
from typing import Optional

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(self.current_context) == 0:
            self.current_context = [[] for _ in range(input_ids.shape[0])]

        # self.current_context.append(input_ids[0][-1].item())
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            _id = input_ids[i][-1].item()
            self.current_context[i].append(_id)
            current_context = self.tokenizer.decode(self.current_context[i])
            should_be_stopped = False
            for word in self.keywords_str:
                if word in current_context:
                    should_be_stopped = True
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)


class KeyWordsCriteriaTrunc(StoppingCriteria):
    def __init__(self, stop_id_sequences, prompt_length):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][self.prompt_length:].tolist()
            should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids.shape[0] == 1:
                    _ids = ids[-len(stop_sequence):]
                else:
                    _ids = ids
                for j in range(len(_ids), 0, -len(stop_sequence)):
                    if _ids[max(j - len(stop_sequence), 0): j] == stop_sequence:
                        should_be_stopped = True
                        break
                if should_be_stopped:
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(model, tuning_embeddings_path, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True, disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    if tuning_embeddings_path:
        prefix_state_dict = torch.load(Path(tuning_embeddings_path) / "prefix_embeddings.pt", map_location=model.device)
        
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            prompt_embeddings = model.get_input_embeddings()(batch_input_ids)
        
        if tuning_embeddings_path:
            print('using prefix embeddings')
            prefix_vector_weights = prefix_state_dict['weight'].to(model.device).to(prompt_embeddings.dtype)

            batch_size = prompt_embeddings.shape[0]
            prefix_embeds = prefix_vector_weights.unsqueeze(0).expand(batch_size, -1, -1)
            input_embeds = torch.cat([prefix_embeds, prompt_embeddings], dim=1)

            prefix_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=model.device)
            new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            generate_input_kwargs = {
                "inputs_embeds": input_embeds,
                "attention_mask": new_attention_mask,
            }

        else:
            print('not using prefix embeddings')
            generate_input_kwargs = {
                "inputs_embeds": prompt_embeddings,
                "attention_mask": attention_mask,
            }

        # try:
        stop_criteria = KeywordsStoppingCriteria(stop_id_sequences, tokenizer)
        batch_outputs = model.generate(
            **generate_input_kwargs,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
            **generation_kwargs
        )


        batch_generations = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        # remove the remain stop sequence from the output.
        for idx, prediction in enumerate(batch_generations):
            for stop_sequence in stop_id_sequences:
                batch_generations[idx] = prediction.split(stop_sequence)[0]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        load_in_8bit=False, 
        load_in_half=True,
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
        use_safetensors=False,
    ):
    import torch 
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, padding_side=padding_side, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, legacy=False, use_fast=use_fast_tokenizer, padding_side=padding_side, trust_remote_code=True)

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("You are using a new tokenizer without a pad token."
                            "This is not supported by this script.")

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.unk_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        # return "", tokenizer
        # defaul load in float16
        # model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        #                                              torch_dtype=torch.float16,
        #                                              device_map=device_map,
        #                                              trust_remote_code=True,
        #                                              use_safetensors=use_safetensors)
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            model = model.cuda()
        if load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer


def _test_generate_completions():
    model_name_or_path = "../models/codellama_7b/v1-16k"
    llm, tokenizer = load_hf_lm_and_tokenizer(
                        model_name_or_path=model_name_or_path, 
                        load_in_half=True,
                        use_fast_tokenizer=True,
                        use_safetensors=True,
                    )
    # some math word problems
    prompts = [
        "---\n1+1=2\n---2+2=4\n---3+3=6\n---4+4=8\n---5+5=10\n---6+6=",
        "---\n1+1=2\n---12+12=24\n---3+3=6\n---12345+12345=",
        # "A train leaves Chicago at 7am and travels at 60mph. Another train leaves Chicago at 9am and travels at 80mph. When will the second train overtake the first?",
        # "The sum of two numbers is 10. The difference of the same two numbers is 4. What are the two numbers?",
    ]

    stop_sequences = ["\n\n\n", "---"]
    # Because many tokenizers will treat the word after space differently from the original word alone, 
    # to be consistent, we add a space before tokenization and remove it after tokenization.
    # stop_id_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    outputs = generate_completions(
            model=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=128,
            batch_size=16,
            # stop_id_sequences=stop_id_sequences,
            stop_id_sequences=stop_sequences,
    )
    print(outputs)

if __name__ == "__main__":
    _test_generate_completions()