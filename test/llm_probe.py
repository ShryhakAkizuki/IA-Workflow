"""Prueba el LLM (DMR) desde Windows
"""
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import urllib.request

from openai import OpenAI

from src.config import settings

base_url = settings.get("LLM_BASE_URL")
api_key = settings.get("LLM_API_KEY") or "no-key"

# 1) GET /models
req = urllib.request.Request(
    base_url.rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {api_key}"},
)

t0 = time.monotonic()

with urllib.request.urlopen(req, timeout=15) as r:
    body = r.read(500).decode("utf-8", "replace")

print(f"[DMR] /models OK en {time.monotonic() - t0:.1f}s -> {body[:200]!r}")

# 2) chat stream corto
client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)
t0 = time.monotonic()
first = None
n = 0

try:
    stream = client.chat.completions.create(
        model=settings.get("LLM_MODEL"),
        messages=[
            {"role": "system", "content": "Responde con una palabra."},
            {"role": "user", "content": "Di hola."},
        ],
        max_tokens=16,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.monotonic() - t0
            n += 1
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()
    print(f"[DMR] stream OK: primer token {first:.2f}s, {n} chunks, total {time.monotonic() - t0:.1f}s")

except Exception as e:
    print(f"[DMR] FALLO en {time.monotonic() - t0:.1f}s: {e!r}")
