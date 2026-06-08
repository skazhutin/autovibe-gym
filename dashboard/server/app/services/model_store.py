"""Model registry persisted to ``data/models.json``."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from gym.model_config import (
    LETOVO_BASE,
    OPENAI_COMPATIBLE_LABEL,
    PROVIDERS,
    TEAM_MODELS,
    provider_uses_base_url,
    runtime_env_for_model,
)


def _seed() -> list[dict[str, Any]]:
    """Seed the registry with the team's gemma + deepseek models on the shared
    Letovo LLM server. Paste the API key once on the Models screen; the
    base/provider are preconfigured."""
    s = get_settings()
    return [
        {
            "id": uuid.uuid4().hex[:8],
            "name": m["name"],
            "provider": OPENAI_COMPATIBLE_LABEL,
            "baseUrl": LETOVO_BASE,
            "apiKeyEnv": "",
            "apiKey": "",
            "ctx": m["ctx"],
            "temp": 0.4,
            "maxTokens": 4096,
            "online": None,
        }
        for m in TEAM_MODELS
    ]


def _load() -> list[dict[str, Any]]:
    s = get_settings()
    if s.models_config.exists():
        try:
            return json.loads(s.models_config.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    seeded = _seed()
    _save(seeded)
    return seeded


def _save(models: list[dict[str, Any]]) -> None:
    s = get_settings()
    s.models_config.write_text(json.dumps(models, indent=2, ensure_ascii=False), "utf-8")


def list_models() -> list[dict[str, Any]]:
    return _load()


def get_model(model_id: str) -> dict[str, Any] | None:
    return next((m for m in _load() if m["id"] == model_id), None)


def create_model(payload: dict[str, Any]) -> dict[str, Any]:
    models = _load()
    record = {
        "id": uuid.uuid4().hex[:8],
        "name": payload["name"],
        "provider": payload.get("provider") or OPENAI_COMPATIBLE_LABEL,
        "baseUrl": payload.get("baseUrl") or "",
        "apiKeyEnv": payload.get("apiKeyEnv") or "",
        "apiKey": payload.get("apiKey") or "",
        "ctx": payload.get("ctx") or 32768,
        "temp": payload.get("temp", 0.4),
        "maxTokens": payload.get("maxTokens") or 8192,
        "online": None,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    models.append(record)
    _save(models)
    return record


def update_model(model_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    models = _load()
    for m in models:
        if m["id"] == model_id:
            for key, value in payload.items():
                if value is not None:
                    m[key] = value
            _save(models)
            return m
    return None


def delete_model(model_id: str) -> bool:
    models = _load()
    kept = [m for m in models if m["id"] != model_id]
    if len(kept) == len(models):
        return False
    _save(kept)
    return True


def _probe(base: str, api_key: str = "") -> dict[str, Any]:
    """Probe an OpenAI-compatible /models endpoint. A reply (incl. 401/403)
    means the server is up; only connection errors mean it is down."""
    base = (base or "").rstrip("/")
    if not base:
        return {"online": False, "error": "no baseUrl"}
    url = base if base.endswith("/models") else base + "/models"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            return {"online": 200 <= resp.status < 500, "status": resp.status}
    except urllib.error.HTTPError as exc:
        return {"online": exc.code in (401, 403, 404), "status": exc.code}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"online": False, "error": str(getattr(exc, "reason", exc))}


def check_health(model_id: str) -> dict[str, Any]:
    model = get_model(model_id)
    if model is None:
        return {"online": False, "error": "model not found"}
    if not provider_uses_base_url(model.get("provider")):
        env = runtime_env_for_model(model)
        configured = bool(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or env.get("LLM_API_KEY"))
        result = {"online": configured, "status": "configured" if configured else "missing api key"}
        _set_online(model_id, configured)
        return result
    api_key_env = model.get("apiKeyEnv") or ""
    api_key = model.get("apiKey") or (os.getenv(api_key_env, "") if api_key_env else "")
    result = _probe(model.get("baseUrl", ""), api_key)
    _set_online(model_id, bool(result.get("online")))
    return result


def server_health() -> dict[str, Any]:
    """Aggregate reachability of the distinct LLM servers in the registry.
    Powers the header 'Сервер онлайн/офлайн' pill."""
    models = _load()
    seen: dict[str, dict[str, Any]] = {}
    for m in models:
        if not provider_uses_base_url(m.get("provider")):
            continue
        base = (m.get("baseUrl") or "").rstrip("/")
        if not base or base in seen:
            continue
        api_key_env = m.get("apiKeyEnv") or ""
        api_key = m.get("apiKey") or (os.getenv(api_key_env, "") if api_key_env else "")
        res = _probe(base, api_key)
        seen[base] = {"baseUrl": base, **res}
    servers = list(seen.values())
    return {
        "online": any(s.get("online") for s in servers),
        "servers": servers,
        "configured": len(servers) > 0,
    }


def _set_online(model_id: str, online: bool) -> None:
    models = _load()
    for m in models:
        if m["id"] == model_id:
            m["online"] = online
    _save(models)
