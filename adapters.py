"""Model adapters: anything the detector can query.

The detector only needs one thing from a model: a callable that maps a list
of prompt strings to a list of response strings. Three adapters are provided.

* :class:`NumpyModelAdapter`  wraps the NumPy transformer from ``model.py``.
* :class:`FunctionAdapter`    wraps any ``str -> str`` Python function.
* :class:`OpenAIChatAdapter`  talks to an OpenAI-compatible chat completions
  endpoint over plain ``urllib`` (no third-party HTTP library).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Protocol, Sequence

from model import Transformer


class TextModel(Protocol):
    """The interface the detector consumes."""

    def __call__(self, prompts: Sequence[str]) -> List[str]: ...


class NumpyModelAdapter:
    """Greedy decoding from the local NumPy transformer.

    ``sep`` is appended to every prompt before generation. It plays the role
    of the turn boundary in a chat template and must match what the model
    saw in training (``plant.SEP``).
    """

    def __init__(self, model: Transformer, sep: str = "\n", max_new_tokens: int = 6):
        self.model = model
        self.sep = sep
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompts: Sequence[str]) -> List[str]:
        return self.model.generate([p + self.sep for p in prompts], max_new_tokens=self.max_new_tokens)


class FunctionAdapter:
    """Wrap a plain ``str -> str`` function."""

    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def __call__(self, prompts: Sequence[str]) -> List[str]:
        return [self.fn(p) for p in prompts]


class OpenAIChatAdapter:
    """Client for ``POST {base_url}/chat/completions`` (OpenAI chat format).

    Works with OpenAI, and with the many local servers and hosted providers
    that speak the same protocol. Requests are sent one at a time with
    temperature 0 so that results are as reproducible as the provider allows.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 3,
        sleep_between: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.sleep_between = sleep_between
        self.n_requests = 0

    def _payload(self, prompt: str) -> dict:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def complete(self, prompt: str) -> str:
        """Send one chat completion request and return the assistant text."""
        body = json.dumps(self._payload(prompt)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.n_requests += 1
                content = data["choices"][0]["message"]["content"]
                return content if isinstance(content, str) else ""
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, TimeoutError) as e:
                last_error = e
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"chat completion failed after {self.max_retries} attempts: {last_error}")

    def __call__(self, prompts: Sequence[str]) -> List[str]:
        out = []
        for p in prompts:
            out.append(self.complete(p))
            if self.sleep_between:
                time.sleep(self.sleep_between)
        return out


def load_local(name: str, out_dir: str = "outputs") -> NumpyModelAdapter:
    """Load ``outputs/{name}.npz`` (``planted`` or ``control``) as a TextModel."""
    path = os.path.join(out_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run plant.py first")
    return NumpyModelAdapter(Transformer.load(path))
