"""
LLM Client to connect to LM Studio's local OpenAI-compatible API.
"""

from typing import Any, Dict, List
import requests
from config import CONFIG
from config.env_loader import get_env


class LMStudioClient:
    def __init__(self):
        self.base_url = get_env("LM_STUDIO_URL", CONFIG["llm"]["base_url"])
        self.model = get_env("LM_STUDIO_MODEL", CONFIG["llm"]["model"])
        self.temperature = CONFIG["llm"]["temperature"]
        self.max_tokens = CONFIG["llm"]["max_tokens"]

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("content", item))))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        return str(content)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=(10, 300)
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"LM Studio API error {response.status_code}: {response.text}"
                )

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return self._normalize_content(content)

        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                "LM Studio took too long to respond. "
                "Try a smaller model, lower max_tokens, or increase the timeout further."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Make sure the server is started."
            )
        except Exception as e:
            raise RuntimeError(f"API failure: {str(e)}")

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
