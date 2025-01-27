from abc import ABC, abstractmethod
import os
import json
from typing import List, Dict, Any
import anthropic
import openai
from dotenv import load_dotenv
import requests

load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response from LLM"""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("Missing ANTHROPIC_API_KEY in environment variables")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        # Convert messages to Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Prepend system message to first user message
                continue
            formatted_messages.append({
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"]
            })
        
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.content[0].text

class DeepseekProvider(LLMProvider):
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY in environment variables")
        self.api_base = "https://api.deepseek.com/v1"  # Update this if needed

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# Model configurations
MODEL_CONFIGS = {
    "gpt-4": {
        "provider": OpenAIProvider,
        "args": {"model": "gpt-4"},
        "description": "OpenAI GPT-4"
    },
    "gpt-3.5-turbo": {
        "provider": OpenAIProvider,
        "args": {"model": "gpt-3.5-turbo"},
        "description": "OpenAI GPT-3.5 Turbo"
    },
    "claude-sonnet": {
        "provider": AnthropicProvider,
        "args": {"model": "claude-3-5-sonnet-latest"},
        "description": "Anthropic Claude 3.5 Sonnet"
    },
    "deepseek-r1": {
        "provider": DeepseekProvider,
        "args": {"model": "deepseek-chat"},
        "description": "DeepSeek Chat R1"
    }
}