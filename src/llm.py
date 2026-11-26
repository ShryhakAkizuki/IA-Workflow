"""Cliente del LLM (Docker Model Runner) con historial y streaming.
"""

import json
import time

from openai import OpenAI

from src.config import settings

client = OpenAI(
    base_url=settings.get("LLM_BASE_URL"),
    api_key=settings.get("LLM_API_KEY") or "no-key",
    timeout=600.0,
)

_messages: list[dict] = [{"role": "system", "content": settings.get("SYSTEM_PROMPT")}]

def _extra_body() -> dict:
    """
    Parsea ``LLM_EXTRA_BODY`` (JSON en una variable de entorno) para mandar
    campos extra en cada request. DMR/llama.cpp usa "chat_template_kwargs".
    """
    raw = (settings.get("LLM_EXTRA_BODY") or "").strip()

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    except json.JSONDecodeError:
        print(f"[LLM] LLM_EXTRA_BODY no es JSON valido, se ignora: {raw!r}")
        return {}

def _trim_history() -> None:
    """Mantiene el system prompt y solo los ``HISTORY_TURNS`` turnos más recientes."""
    max_msgs = settings.history_turns * 2
    overflow = len(_messages) - 1 - max_msgs

    if overflow > 0:
        del _messages[1:1 + overflow]

def chat_stream(user_text: str):
    """Envía el texto del usuario y cede fragmentos de la respuesta en vivo.
    """
    _messages.append({"role": "user", "content": user_text})
    _trim_history()

    parts: list[str] = []
    stream = client.chat.completions.create(
        model=settings.get("LLM_MODEL"),
        messages=_messages,
        temperature=settings.llm_temperature,
        stream=True,
        extra_body=_extra_body(),
    )

    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                parts.append(delta.content)
                yield delta.content
    finally:
        full = "".join(parts).strip()
        if full:
            _messages.append({"role": "assistant", "content": full})
            _trim_history()


_WARMUP_RETRIES = 10
_WARMUP_BACKOFF = 2.0


def llm_warmup() -> bool:
    """
    Pre-carga el modelo en DMR para que el primer mensaje real sea rapido.
    Devuelve True si el warmup termino OK; False si el servicio no respondio.
    """
    import urllib.request

    base = settings.get("LLM_BASE_URL").rstrip("/")
    api_key = settings.get("LLM_API_KEY") or "no-key"

    up = False
    for attempt in range(1, _WARMUP_RETRIES + 1):
        try:
            req = urllib.request.Request(
                base + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            up = True
            break
        except Exception:
            wait = _WARMUP_BACKOFF * attempt
            print(f"[LLM] warmup: runner no disponible (intento {attempt}/{_WARMUP_RETRIES}), reintentando en {wait:.0f}s...")
            time.sleep(wait)

    if not up:
        print("[LLM] warmup: el runner no respondio; la primera frase puede ser lenta")
        return False

    t0 = time.perf_counter()
    try:
        client.chat.completions.create(
            model=settings.get("LLM_MODEL"),
            messages=[
                {"role": "user", "content": "Di: listo"},
            ],
            max_tokens=4,
            temperature=0.0,
            stream=False,
            extra_body=_extra_body(),
        )
        print(f"[LLM] warmup OK: modelo precargado en {time.perf_counter() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"[LLM] warmup fallido en {time.perf_counter() - t0:.1f}s: {e}")
        return False
