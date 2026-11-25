r"""Prueba rapida del pipeline: STT (ScribeCrate WS) + LLM (DMR).

Uso (desde la raiz del repo):
    python test/test_pipeline.py
"""
import sys
import time
from pathlib import Path

# Anade la raiz del repo al sys.path para poder importar el paquete `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.llm import chat_stream
from src.stt import transcribe_wav


def main() -> int:
    print("=" * 60)
    print("TEST PIPELINE STT + LLM")
    print("=" * 60)
    print(
        "STT: "
        f"transport={settings.get('STT_TRANSPORT')} "
        f"model={settings.get('STT_MODEL')}"
    )
    print(
        f"     ws={settings.get('STT_WS_URL')} "
        f"rest={settings.get('STT_REST_URL')}"
    )
    print(f"     api_key={'<setada>' if settings.get('STT_API_KEY') else '<SIN KEY>'}")
    print(f"LLM: {settings.get('LLM_MODEL')}")
    print(f"     @ {settings.get('LLM_BASE_URL')}")
    print("-" * 60)

    # --- 1) STT ---
    wav_path = Path(__file__).resolve().parent / "test_es.wav"
    t0 = time.monotonic()
    text = transcribe_wav(str(wav_path))
    dt_stt = time.monotonic() - t0
    print(f"\n[STT] transcripcion ({dt_stt:.1f}s):")
    print(f"  > {text!r}")
    if not text:
        print("[STT] FALLO: sin texto")
        return 1

    # --- 2) LLM ---
    prompt = (
        "Responde en una sola frase muy breve confirmando que entiendes "
        f"esta transcripcion: «{text}»"
    )
    t0 = time.monotonic()
    first = None
    out = []
    for chunk in chat_stream(prompt):
        if first is None:
            first = time.monotonic() - t0
        out.append(chunk)
        print(chunk, end="", flush=True)
    dt_llm = time.monotonic() - t0
    print(f"\n[LLM] OK — primer token {first:.2f}s, total {dt_llm:.2f}s")
    print("\n" + "=" * 60)
    print("PIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
