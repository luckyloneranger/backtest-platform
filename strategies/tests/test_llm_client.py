"""Tests for Azure OpenAI client wrapper."""

import json
from unittest.mock import patch, MagicMock

import pytest

from strategies.llm_client import AzureOpenAIClient, LLMClientError


def test_missing_env_vars_raises():
    """Client raises LLMClientError when env vars are not set."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_ENDPOINT"):
            AzureOpenAIClient()


def test_missing_api_key_raises():
    """Client raises LLMClientError when API key is missing."""
    with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"}, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_API_KEY"):
            AzureOpenAIClient()


def test_missing_deployment_raises():
    """Client raises LLMClientError when deployment is missing."""
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
    }
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(LLMClientError, match="AZURE_OPENAI_DEPLOYMENT"):
            AzureOpenAIClient()


def test_client_init_from_env():
    """Client initializes correctly from env vars."""
    env = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }
    with patch.dict("os.environ", env, clear=True):
        client = AzureOpenAIClient()
        assert client.endpoint == "https://test.openai.azure.com"
        assert client.deployment == "gpt-4o"


def test_client_init_explicit_params():
    """Client accepts explicit params over env vars."""
    client = AzureOpenAIClient(
        endpoint="https://custom.openai.azure.com",
        api_key="custom-key",
        deployment="gpt-4o-mini",
    )
    assert client.endpoint == "https://custom.openai.azure.com"
    assert client.deployment == "gpt-4o-mini"


def test_chat_completion_success():
    """Successful API call returns assistant message content."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '[{"action": "BUY", "symbol": "RELIANCE", "quantity": 10}]'}}]
    }

    with patch("strategies.llm_client.requests.post", return_value=mock_response) as mock_post:
        result = client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=512,
        )
        assert result == '[{"action": "BUY", "symbol": "RELIANCE", "quantity": 10}]'
        mock_post.assert_called_once()


def test_chat_completion_auth_error():
    """401 raises LLMClientError immediately (no retry)."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="bad-key",
        deployment="gpt-4o",
    )
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

    with patch("strategies.llm_client.requests.post", return_value=mock_response):
        with pytest.raises(LLMClientError, match="Authentication failed"):
            client.chat_completion(messages=[{"role": "user", "content": "test"}])


def test_chat_completion_retry_on_429():
    """429 triggers retry, succeeds on second attempt."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
    )

    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limited"

    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "choices": [{"message": {"content": "[]"}}]
    }

    with patch("strategies.llm_client.requests.post", side_effect=[rate_limit_response, success_response]):
        with patch("time.sleep"):  # skip actual sleep
            result = client.chat_completion(messages=[{"role": "user", "content": "test"}])
            assert result == "[]"


def test_chat_completion_max_retries_exceeded():
    """Exhausting retries raises LLMClientError."""
    client = AzureOpenAIClient(
        endpoint="https://test.openai.azure.com",
        api_key="test-key",
        deployment="gpt-4o",
        max_retries=2,
    )

    error_response = MagicMock()
    error_response.status_code = 500
    error_response.text = "Internal Server Error"

    with patch("strategies.llm_client.requests.post", return_value=error_response):
        with patch("time.sleep"):
            with pytest.raises(LLMClientError, match="after 2 retries"):
                client.chat_completion(messages=[{"role": "user", "content": "test"}])
