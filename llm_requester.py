import time
from typing import List
import anthropic
from openai import OpenAI
import os
import requests
import json
from dotenv import load_dotenv
# from vllm import LLM, SamplingParams
load_dotenv()
# from swebench import generate_test_cases_for_swebench
# import vertexai
# from vertexai.generative_models._generative_models import ResponseValidationError
# from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig
# Import necessary libraries for different LLMs
from abc import ABC, abstractmethod
from llamaapi import LlamaAPI
# import google.generativeai as genai
from dataclasses import dataclass
backends = ['openAI', 'fireworks', 'VLLM']

class LLMRequester(ABC):
    @abstractmethod
    def get_completion(self, messages, **kwargs):
        raise NotImplementedError()
    def get_total_usage(self):
        raise NotImplementedError()
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other):
        return TokenUsage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens, self.total_tokens + other.total_tokens)

class FireworksAPIRequester(LLMRequester):
    def __init__(self, name, token_usage:TokenUsage):
        self.name = name
        self.key = os.getenv("fireworks_key")
        self.token_usage = token_usage

    def get_total_usage(self):
        return self.token_usage

    def get_completion(self, messages, **kwargs) -> list[str]:

        prompt = ''.join([message['content'] for message in messages])
        # print(prompt)
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": f"accounts/fireworks/models/{self.name}",
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": kwargs["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "n":kwargs['n'],
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
        }
        try:
            res = requests.request("POST", url, headers=headers, data=json.dumps(payload))
            res = res.json()
            self.token_usage.completion_tokens += res['usage']['completion_tokens']
            self.token_usage.prompt_tokens += res['usage']['prompt_tokens']
            self.token_usage.total_tokens += res['usage']['total_tokens']
            return [res['message']['content'] for res in res['choices']]
        except (requests.exceptions.JSONDecodeError, KeyError, Exception) as e:
            print(res)
            print('Retrying after 10 seconds ...')
            return self.get_completion(messages, **kwargs)
            return [prompt]


class OpenaiRequester(LLMRequester):
    def __init__(self, name,token_usage, backend=None):
        self.key = os.getenv('openai_key')
        self.token_usage = token_usage
        if backend is None:
            self.client = OpenAI(api_key=self.key)
        else:
            if backend == "https://api.aimlapi.com/v1":
                self.key = os.getenv('aimlapi_key')
            elif backend == "https://api.deepinfra.com/v1/openai":
                self.key = os.getenv('deepinfraapi_key')
            else:
                self.key = os.getenv('deepseek_key')
            self.client = OpenAI(api_key=self.key, base_url=backend)
        self.name = name

    def get_total_usage(self):
        return self.token_usage

    def get_completion(self,
            messages: list[str],
            temperature=0,
            seed=123,
            n=1,
    ) -> list[str]:
        messages[0]['content'] = messages[0]['content'] + "PUT THE PYTHON IMPLEMENTATION IN BETWEEN ```python and ``` tags "
        params = {
            "model": self.name,
            "messages": messages,
            "seed": seed,
            "n": n
        }
        if self.name not in ('gpt-5-mini', 'o3-mini', 'o3'):
            params['temperature'] = temperature
        try:
            completion = self.client.chat.completions.create(**params)
        except Exception as e:
            print(e)
            raise e
        self.token_usage.completion_tokens += completion.usage.completion_tokens
        self.token_usage.prompt_tokens += completion.usage.prompt_tokens
        self.token_usage.total_tokens += completion.usage.total_tokens
        return [choice.message.content for choice in completion.choices]


import os
from typing import List, Dict, Any


class VLLMRequester(LLMRequester):
    """
    Local inference using vLLM with Hugging Face models.
    Example model names: "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", etc.

    Optional env vars:
      VLLM_TP_SIZE                -> tensor parallel size (default: 1)
      VLLM_GPU_MEM_UTIL           -> gpu_memory_utilization (default: 0.9)
      VLLM_DTYPE                  -> dtype, e.g. "auto", "float16", "bfloat16" (default: "auto")
      VLLM_TRUST_REMOTE_CODE      -> "1" to enable trust_remote_code (default: "0")
    """

    def __init__(
        self,
        name: str,
        token_usage,
        *,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float | None = None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        **llm_kwargs: Any,
    ):
        self.name = name
        self.token_usage = token_usage
        if name == 'gemma3':
            path_to_gemma3 = "/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face/models--google--gemma-3-12b-it"
        else:
            raise NotImplementedError("The model should be downloaded in the local")
        # Defaults can also be provided through env vars
        tp = int(os.getenv("VLLM_TP_SIZE", "1")) if tensor_parallel_size is None else tensor_parallel_size
        gmu = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.9")) if gpu_memory_utilization is None else gpu_memory_utilization
        dt = os.getenv("VLLM_DTYPE", "auto") if dtype is None else dtype
        trc = (os.getenv("VLLM_TRUST_REMOTE_CODE", "0") == "1") if trust_remote_code is None else trust_remote_code

        # Spin up a local vLLM engine for the HF model
        # You can pass extra engine params via **llm_kwargs (e.g. max_model_len, max_num_seqs, enforce_eager)
        self.llm = LLM(
            model=path_to_gemma3,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gmu,
            dtype=dt,
            trust_remote_code=trc,
            **llm_kwargs,
        )

        # Cache tokenizer for chat templating & token counting
        self.tokenizer = self.llm.get_tokenizer()

    def get_total_usage(self):
        return self.token_usage

    def _to_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages into a single prompt using the model's HF chat template.
        """
        # HF expects [{"role":"system"/"user"/"assistant", "content":"..."}] shape
        # Ensure generation prompt is appended
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt

    def _count_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        return len(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        )

    def get_completion(
        self,
        messages: list[dict],
        temperature: float = 0,
        seed: int = 123,
        n: int = 1,
    ) -> list[str]:
        # keep your prompt tweak on the first message
        messages[0]["content"] = (
            messages[0]["content"]
            + "PUT THE PYTHON IMPLEMENTATION IN BETWEEN ```python and ``` tags "
        )

        # Build prompt via HF chat template
        prompt = self._to_prompt_from_messages(messages)

        # Sampling parameters (vLLM follows OpenAI-style knobs; `seed` is supported) :contentReference[oaicite:3]{index=3}
        sampling = SamplingParams(
            temperature=temperature,
            n=n,
            seed=seed,
            # You can expose more knobs here if you want parity with your OpenAI path:
            # top_p=..., top_k=..., max_tokens=..., repetition_penalty=..., presence_penalty=..., frequency_penalty=...
        )

        # Generate (single prompt, possibly multiple candidates via n)
        try:
            outputs = self.llm.generate([prompt], sampling)
        except Exception as e:
            print(e)
            raise e

        # vLLM returns a list of RequestOutput; we only passed one prompt
        req_out = outputs[0]

        # Token accounting: prompt tokens via tokenizer; completion tokens via each choice's token_ids
        # (vLLM doesn’t always return aggregated usage like OpenAI; we compute it here.) :contentReference[oaicite:4]{index=4}
        prompt_tokens = self._count_prompt_tokens(messages)
        completion_tokens_total = 0
        texts: list[str] = []

        for choice in req_out.outputs:
            texts.append(choice.text)
            if hasattr(choice, "token_ids") and choice.token_ids is not None:
                completion_tokens_total += len(choice.token_ids)

        # Update your shared token_usage object defensively
        self.token_usage.prompt_tokens += prompt_tokens
        self.token_usage.completion_tokens += completion_tokens_total
        self.token_usage.total_tokens += prompt_tokens + completion_tokens_total

        return texts


def init_llm(model: str, backend:str) -> LLMRequester:
    token_usage = TokenUsage()
    if backend == 'openAI':
        llm = OpenaiRequester(name=model, token_usage=token_usage)
    elif backend == 'fireworks':
        llm = FireworksAPIRequester(name=model,token_usage=token_usage)
    elif backend == "VLLM":
        llm = VLLMRequester(name=model,token_usage=token_usage)
    else:
        raise ValueError('backend not known')
    return llm