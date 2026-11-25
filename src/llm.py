"""Cliente del LLM (Docker Model Runner) con historial y streaming.
"""

import json

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
