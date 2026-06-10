from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_PATH = REPO_ROOT / "dashboard" / "server" / "data" / "models.json"

OPENAI_COMPATIBLE_LABEL = "OpenAI-совместимый"
VLLM_LABEL = "vLLM"
GEMINI_LABEL = "Gemini"
LITELLM_LABEL = "LiteLLM"

PROVIDERS = [OPENAI_COMPATIBLE_LABEL, VLLM_LABEL, GEMINI_LABEL, LITELLM_LABEL]
LETOVO_BASE = "http://llm.letovo.site:8809/openai"
LEGACY_LLM_KEY_ENVS = {
    "LLM_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
}

TEAM_MODELS = [
    {"name": "gemma-4-26b", "ctx": 32768},
    {"name": "deepseek-v4-flash", "ctx": 65536},
]


def registry_path() -> Path:
    return Path(os.getenv("AUTOVIBE_MODELS_CONFIG", str(DEFAULT_MODELS_PATH)))


def normalize_provider(provider: str | None) -> str:
    raw = (provider or OPENAI_COMPATIBLE_LABEL).strip().lower()
    if "gemini" in raw or "google" in raw:
        return "google"
    if "litellm" in raw or "lite" in raw:
        return "litellm"
    if "vllm" in raw:
        return "vllm"
    return "openai"


def provider_label(provider: str | None) -> str:
    kind = normalize_provider(provider)
    if kind == "google":
        return GEMINI_LABEL
    if kind == "litellm":
        return LITELLM_LABEL
    if kind == "vllm":
        return VLLM_LABEL
    return OPENAI_COMPATIBLE_LABEL


def provider_uses_base_url(provider: str | None) -> bool:
    return normalize_provider(provider) in {"openai", "vllm"}


def _default_record(name: str, *, ctx: int) -> dict[str, Any]:
    return {
        "name": name,
        "provider": OPENAI_COMPATIBLE_LABEL,
        "baseUrl": LETOVO_BASE,
        "apiKeyEnv": "",
        "apiKey": "",
        "ctx": ctx,
        "temp": 0.4,
        "maxTokens": 4096,
        "online": None,
    }


def seed_records() -> list[dict[str, Any]]:
    import uuid

    records = []
    for model in TEAM_MODELS:
        record = _default_record(model["name"], ctx=model["ctx"])
        record["id"] = uuid.uuid4().hex[:8]
        records.append(record)
    return records


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or registry_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text("utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    for model in data:
        if isinstance(model, dict) and model.get("apiKeyEnv") in LEGACY_LLM_KEY_ENVS:
            model["apiKeyEnv"] = ""
    return data


def save_registry(models: list[dict[str, Any]], path: Path | None = None) -> None:
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(models, indent=2, ensure_ascii=False), "utf-8")


def find_model(ref: str | None, models: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if not ref:
        return None
    needle = str(ref).strip()
    for model in models if models is not None else load_registry():
        if str(model.get("id", "")) == needle or str(model.get("name", "")) == needle:
            return model
    return None


def runtime_env_for_model(model: dict[str, Any]) -> dict[str, str]:
    kind = normalize_provider(model.get("provider"))
    out = {"LLM_PROVIDER": "openai" if kind == "vllm" else kind}
    if model.get("name"):
        out["LLM_MODEL"] = str(model["name"])

    if provider_uses_base_url(model.get("provider")) and model.get("baseUrl"):
        out["LLM_BASE_URL"] = str(model["baseUrl"])
    if model.get("ctx"):
        out["AUTOVIBE_CTX_LIMIT"] = str(int(model["ctx"]))
    if model.get("maxTokens"):
        out["AUTOVIBE_MAX_TOKENS_LIMIT"] = str(int(model["maxTokens"]))

    key_env = model.get("apiKeyEnv") or ""
    key = model.get("apiKey") or (os.getenv(key_env, "") if key_env else "")
    if key:
        if kind == "google":
            out["GEMINI_API_KEY"] = str(key)
            out["GOOGLE_API_KEY"] = str(key)
        elif kind == "litellm":
            out["AUTOVIBE_LITELLM_API_KEY"] = str(key)
        else:
            out["LLM_API_KEY"] = str(key)
    return out


def apply_model_reference(ref: str | None) -> str:
    model = find_model(ref)
    if not model:
        raise ValueError(
            f"Model {ref!r} is not configured. Add it with "
            "`python -m experiments.models add ...` or from the dashboard Models page."
        )
    for key, value in runtime_env_for_model(model).items():
        os.environ[key] = value
    return str(model.get("name") or ref)
