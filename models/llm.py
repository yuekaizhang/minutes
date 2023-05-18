
import os
import torch
from typing import Any, List, Mapping, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI

def load_llm(llm_name_or_path):
    if llm_name_or_path is None or 'openai' in llm_name_or_path.lower():
        return ChatOpenAI()
    elif 'chatglm' in llm_name_or_path.lower():
        return ChatGLM(llm_name_or_path)
    else:
        return AutoLLM(llm_name_or_path)

def get_model_and_tokenizer(model_path: str, load_in_8bit: bool = False):

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    if 'chatglm' in model_path:
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, config=model_config, trust_remote_code=True, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = model.half().cuda()
    elif 'openbuddy' in model_path or 'lama' in model_path:
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map) 
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = model.half().cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, load_in_8bit=load_in_8bit, trust_remote_code=True, device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

class AutoLLM(LLM):
    model: object = None
    tokenizer: object = None

    def __init__(self, model_name_or_path: str, load_in_8bit: bool = False):
        super().__init__()
        self.load_model(model_name_or_path, load_in_8bit)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def load_model(self, model_name_or_path: str, load_in_8bit: bool = False):
        self.model, self.tokenizer = get_model_and_tokenizer(model_name_or_path, load_in_8bit)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: float = 0.1,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        do_sample: bool = False,
        max_new_tokens: int = 512,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=False,
                output_scores=False,
            )
            outputs = generation_output[0][len(input_ids[0]):]
            sentence = self.tokenizer.decode(outputs, skip_special_tokens=True)

        return sentence

class ChatGLM(LLM):
    model: object = None
    tokenizer: object = None

    def __init__(self, model_name_or_path: str, load_in_8bit: bool = False):
        super().__init__()
        self.load_model(model_name_or_path, load_in_8bit)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def load_model(self, model_name_or_path: str, load_in_8bit: bool = False):
        self.model, self.tokenizer = get_model_and_tokenizer(model_name_or_path, load_in_8bit)

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              temperature: float = 0.1,
              top_p: float = 0.75,
              top_k: int = 40,
              max_length: int = 4096,
              max_new_tokens: int = 512,
              streaming: bool = False):
        if streaming:
            raise ValueError("streaming is not permitted.")
        else:
            sentence, _ = self.model.chat(
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            return sentence