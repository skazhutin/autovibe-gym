"""Manage the shared LLM model registry used by the dashboard and CLI runs."""
from __future__ import annotations

import argparse
import json
import uuid

from gym.model_config import (
    OPENAI_COMPATIBLE_LABEL,
    PROVIDERS,
    find_model,
    load_registry,
    provider_label,
    provider_uses_base_url,
    save_registry,
)


def _record_from_args(args: argparse.Namespace) -> dict:
    provider = provider_label(args.provider)
    if provider_uses_base_url(provider) and not args.base_url:
        raise SystemExit(f"{provider} requires --base-url")
    return {
        "id": args.id or uuid.uuid4().hex[:8],
        "name": args.name,
        "provider": provider,
        "baseUrl": args.base_url or "",
        "apiKeyEnv": args.api_key_env or "",
        "apiKey": args.api_key or "",
        "ctx": args.ctx,
        "temp": args.temp,
        "maxTokens": args.max_tokens,
        "online": None,
    }


def _print_table(models: list[dict]) -> None:
    if not models:
        print("No models configured.")
        return
    rows = [
        (
            str(m.get("id", "")),
            str(m.get("name", "")),
            str(m.get("provider", "")),
            str(m.get("baseUrl", "")) or "-",
            str(m.get("apiKeyEnv", "")) or "-",
        )
        for m in models
    ]
    headers = ("id", "name", "provider", "baseUrl", "apiKeyEnv")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage AutoVibe Gym LLM models.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List configured models.")

    show = sub.add_parser("show", help="Show one model as JSON.")
    show.add_argument("ref", help="Model id or name.")

    add = sub.add_parser("add", help="Add a model.")
    add.add_argument("--id", default=None)
    add.add_argument("--name", required=True)
    add.add_argument("--provider", choices=PROVIDERS + ["openai", "google", "gemini", "litellm"], default=OPENAI_COMPATIBLE_LABEL)
    add.add_argument("--base-url", default="")
    add.add_argument("--api-key-env", default=None)
    add.add_argument("--api-key", default="")
    add.add_argument("--ctx", type=int, default=32768)
    add.add_argument("--temp", type=float, default=0.4)
    add.add_argument("--max-tokens", type=int, default=8192)

    delete = sub.add_parser("delete", help="Delete a model by id or name.")
    delete.add_argument("ref")

    args = parser.parse_args()
    models = load_registry()

    if args.cmd == "list":
        _print_table(models)
        return

    if args.cmd == "show":
        model = find_model(args.ref, models)
        if model is None:
            raise SystemExit(f"Model not found: {args.ref}")
        safe = dict(model)
        if safe.get("apiKey"):
            safe["apiKey"] = "********"
        print(json.dumps(safe, indent=2, ensure_ascii=False))
        return

    if args.cmd == "add":
        if find_model(args.name, models) or (args.id and find_model(args.id, models)):
            raise SystemExit(f"Model already exists: {args.id or args.name}")
        record = _record_from_args(args)
        models.append(record)
        save_registry(models)
        print(f"Added {record['id']} {record['name']} ({record['provider']})")
        return

    if args.cmd == "delete":
        model = find_model(args.ref, models)
        if model is None:
            raise SystemExit(f"Model not found: {args.ref}")
        kept = [m for m in models if m is not model]
        save_registry(kept)
        print(f"Deleted {model.get('id')} {model.get('name')}")


if __name__ == "__main__":
    main()
