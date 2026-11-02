"""One shared, explicitly configured OpenAI-compatible HTTP adapter. Owner: B.

No default provider, mock fallback, or network call is made at import time.
"""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from evrag.config import ROOT, load_config


class LLMConfigurationError(RuntimeError):
    """The team has not configured a usable model, endpoint, or credential."""


class LLMCallError(RuntimeError):
    """The endpoint failed or returned an invalid completion."""


def generate(prompt: str, *, config=None, transport=None) -> str:
    """Return one answer, using LLM_* environment settings or shared YAML.

    ``transport`` is an optional urllib-compatible opener used for offline tests.
    Errors propagate rather than being counted as model abstentions. There are
    deliberately no hidden paid retries. Providers need chat/completions support.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass  # Exported environment variables still work without this helper.
    else:
        load_dotenv(ROOT / ".env", override=False)
    settings = (load_config() if config is None else config)["llm"]
    key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip() or settings.get("model")
    base = os.getenv("LLM_BASE_URL", "").strip() or settings.get("base_url")
    if not key or not model or model == "TODO" or not base:
        raise LLMConfigurationError(
            "Set LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL after the team chooses a provider"
        )
    parsed = urlsplit(base)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LLMConfigurationError("LLM_BASE_URL must not contain credentials, queries, or fragments")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise LLMConfigurationError("Use HTTPS for LLM_BASE_URL (HTTP is allowed only on localhost)")
    timeout = float(settings.get("timeout_seconds", 30))
    if timeout <= 0:
        raise LLMConfigurationError("timeout_seconds must be positive")
    payload = {"model": model, "temperature": settings["temperature"],
               "max_tokens": settings["max_tokens"],
               "messages": [{"role": "user", "content": prompt}]}
    request = Request(base.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
    try:
        with (urlopen if transport is None else transport)(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        answer = result["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing answer text")
        return answer.strip()
    except HTTPError as exc:
        # Do not include the response body, headers, key, or prompt in errors.
        raise LLMCallError(f"LLM endpoint returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise LLMCallError("LLM endpoint connection failed or timed out") from None
    except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
        raise LLMCallError("LLM endpoint returned a malformed or empty completion") from None
