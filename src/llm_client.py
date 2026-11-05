"""
Ollama LLM Client Module
Provides a robust, local HTTP client to communicate with the local Ollama instance (localhost:11434).
Supports connection health checks, model discovery, model pulling, non-streaming and streaming generation.
"""

from typing import Any, Dict, Generator, List, Optional
import json
import requests


class OllamaClient:
    """
    Communicates with Ollama running locally at http://localhost:11434.
    Adheres strictly to local execution requirements without external APIs.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """
        Verifies whether the Ollama local daemon is reachable.
        
        Returns:
            Dict with 'status': bool, and 'message': str.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                return {
                    "status": True,
                    "message": "Ollama is running locally.",
                    "models": models
                }
            return {
                "status": False,
                "message": f"Ollama returned HTTP status {resp.status_code}.",
                "models": []
            }
        except requests.ConnectionError:
            return {
                "status": False,
                "message": (
                    "Could not connect to Ollama at http://localhost:11434. "
                    "Please ensure Ollama is installed and running (run 'ollama serve' or start Ollama)."
                ),
                "models": []
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Error communicating with Ollama: {str(e)}",
                "models": []
            }

    def list_models(self) -> List[str]:
        """Retrieves list of installed models from local Ollama instance."""
        health = self.check_health()
        if health["status"]:
            return health.get("models", [])
        return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,
        stream: bool = False
    ) -> str:
        """
        Sends a generation request to the Ollama /api/generate endpoint.
        
        Args:
            prompt: Formatted user prompt.
            system_prompt: System instructions / grounding prompt.
            model: Ollama model tag (e.g. 'llama3.2:3b', 'mistral', etc.).
            temperature: Sampling temperature (0.0 to 1.0).
            stream: Whether to stream tokens.
            
        Returns:
            Generated text response string.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to local Ollama server at {self.base_url}. "
                "Ensure Ollama is started."
            ) from e
        except requests.HTTPError as e:
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' was not found in Ollama. "
                    f"Please run 'ollama run {model}' or 'ollama pull {model}' to download it."
                ) from e
            raise RuntimeError(f"Ollama generation failed with status {resp.status_code}: {resp.text}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during generation: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "llama3.2:3b",
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        """
        Streams generated tokens one by one as they arrive from Ollama.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with requests.post(endpoint, json=payload, stream=True, timeout=self.timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        chunk_data = json.loads(line)
                        token = chunk_data.get("response", "")
                        yield token
                        if chunk_data.get("done", False):
                            break
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Streaming error: {str(e)}") from e
