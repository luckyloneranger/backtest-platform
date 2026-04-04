"""Azure OpenAI client wrapper for LLM-based strategies."""

import os
import time

import requests


class LLMClientError(Exception):
    """Raised on Azure OpenAI client errors (auth, config, retries exhausted)."""


class AzureOpenAIClient:
    """Thin wrapper around Azure OpenAI chat completions REST API.

    Config priority: explicit params > env vars.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-02-01",
        max_retries: int = 3,
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise LLMClientError("AZURE_OPENAI_ENDPOINT not set")

        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self._api_key:
            raise LLMClientError("AZURE_OPENAI_API_KEY not set")

        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise LLMClientError("AZURE_OPENAI_DEPLOYMENT not set")

        self.api_version = api_version
        self.max_retries = max_retries

        # Strip trailing slash from endpoint
        self.endpoint = self.endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat completion request and return the assistant's message content.

        Retries on 429 (rate limit) and 5xx (server errors) with exponential backoff.
        Raises LLMClientError on auth errors or after exhausting retries.
        """
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(self.max_retries):
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            if response.status_code == 401:
                raise LLMClientError(f"Authentication failed: {response.text}")

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text}"
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            raise LLMClientError(f"Unexpected HTTP {response.status_code}: {response.text}")

        raise LLMClientError(f"Request failed after {self.max_retries} retries: {last_error}")
