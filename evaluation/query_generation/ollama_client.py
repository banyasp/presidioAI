"""Ollama API client for query generation using local LLMs."""

import requests
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
        """
        self.base_url = base_url
        self.model = model

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(self, prompt: str, max_retries: int = 3, timeout: int = 60) -> Optional[str]:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Generated text or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                        }
                    },
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        logger.error(f"Failed to generate after {max_retries} attempts")
        return None


def call_ollama(prompt: str, model: str = "llama3.2:1b") -> Optional[str]:
    """
    Convenience function to call Ollama.

    Args:
        prompt: Input prompt
        model: Model name

    Returns:
        Generated text or None if failed
    """
    client = OllamaClient(model=model)

    if not client.is_available():
        logger.error("Ollama is not running. Start it with: ollama serve")
        return None

    available_models = client.list_models()
    if model not in available_models:
        logger.error(f"Model {model} not found. Available models: {available_models}")
        logger.info(f"Install with: ollama pull {model}")
        return None

    return client.generate(prompt)


if __name__ == "__main__":
    # Test the client
    client = OllamaClient()

    if not client.is_available():
        print("ERROR: Ollama is not running")
        print("Start Ollama with: ollama serve")
        exit(1)

    print(f"✓ Ollama is running")
    print(f"Available models: {client.list_models()}")

    # Test generation
    test_prompt = "What is the capital of France?"
    print(f"\nTest prompt: {test_prompt}")
    result = client.generate(test_prompt)
    print(f"Response: {result}")
