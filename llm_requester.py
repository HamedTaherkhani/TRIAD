import time
from typing import List
import anthropic
from openai import OpenAI
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()
from abc import ABC, abstractmethod
from llamaapi import LlamaAPI
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
        if backend == 'VLLM':
            backend = "http://localhost:8000/v1"
            self.key = 'EMPTY'
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


def init_llm(model: str, backend:str) -> LLMRequester:
    token_usage = TokenUsage()
    if backend == 'openAI':
        llm = OpenaiRequester(name=model, token_usage=token_usage)
    elif backend == 'fireworks':
        llm = FireworksAPIRequester(name=model,token_usage=token_usage)
    elif backend == "VLLM":
        llm = OpenaiRequester(name=model,token_usage=token_usage, backend=backend)
    else:
        raise ValueError('backend not known')
    return llm