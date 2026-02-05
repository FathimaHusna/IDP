from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """Lightweight provider-agnostic chat client.

    Configure via env:
    - LLM_ENABLED=1
    - LLM_PROVIDER=openai|azure|open-router|local
    - OPENAI_API_KEY, OPENAI_MODEL (e.g., gpt-4o-mini)
    - OPENAI_BASE_URL (optional; defaults to https://api.openai.com/v1)
    - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    - LOCAL_LLM_BASE_URL (e.g., http://localhost:11434/v1)
    """

    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    @property
    def enabled(self) -> bool:
        return os.environ.get("LLM_ENABLED", "0") == "1"

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs: Any) -> str:
        if not self.enabled:
            raise RuntimeError("LLM is disabled. Set LLM_ENABLED=1 and provider env vars.")
        if self.provider == "openai" or self.provider == "open-router":
            return self._chat_openai(messages, model=model, **kwargs)
        if self.provider == "azure":
            return self._chat_azure(messages, model=model, **kwargs)
        if self.provider == "local":
            return self._chat_local(messages, model=model, **kwargs)
        raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _chat_openai(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs: Any) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0)}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _chat_azure(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs: Any) -> str:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not (endpoint and api_key and deployment):
            raise RuntimeError("Azure OpenAI env vars not set")
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        payload = {"messages": messages, "temperature": kwargs.get("temperature", 0)}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _chat_local(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs: Any) -> str:
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        model = model or os.environ.get("LOCAL_LLM_MODEL", "llama3")
        payload = {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0)}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from a string."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None
    return None

