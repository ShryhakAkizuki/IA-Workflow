"""Orquestador: micrófono -> ScribeCrate Live (STT) -> Docker Model Runner (LLM) -> consola.

Al arrancar se precarga el LLM (DMR) y la sesion WebSocket de STT para que
la primera frase no pague el costo de carga del modelo.

Uso (desde la raiz del repo):
    python -m src.main        (Ctrl+C para salir)
"""
import sys
import time
from pathlib import Path
from threading import Thread

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sounddevice as sd

from src.audio import _input_device, record_until_silence
from src.config import settings
from src.llm import chat_stream, llm_warmup
from src.stt import stt_close, stt_warmup, transcribe_audio

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _timed_chat(user_text: str) -> tuple[float, float]:
    """Ejecuta la respuesta en streaming; devuelve (ttft_s, total_llm_s)."""
    t_llm_start = time.perf_counter()
    t_first = None
    t_last = t_llm_start
    print("  Asistente: ", end="", flush=True)

    for chunk in chat_stream(user_text):
        if t_first is None:
            t_first = time.perf_counter()
        print(chunk, end="", flush=True)
        t_last = time.perf_counter()

    print()
    ttft = (t_first - t_llm_start) if t_first is not None else 0.0
    total = t_last - t_llm_start

    return ttft, total

def main() -> None:
    print(
        "Orquestador STT+LLM | "
        f"STT: ScribeCrate [{settings.get('STT_TRANSPORT')}, {settings.get('STT_MODEL')}] | "
        f"LLM: {settings.get('LLM_BASE_URL')} | Ctrl+C para salir."
    )
    try:
        dev = _input_device()
        print(f"Dispositivo de entrada: {sd.query_devices(dev)}")
    except Exception as e:
        print(f"Aviso: no se pudo listar el micro ({e})")

    print("Precargando modelos (LLM + STT)...")
    Thread(target=llm_warmup, daemon=True).start()
    Thread(target=stt_warmup, daemon=True).start()

    try:
        while True:
            try:
                print("\nEscuchando...")
                audio = record_until_silence()
                if audio is None or len(audio) == 0:
                    continue

                t_stt_start = time.perf_counter()
                text = transcribe_audio(audio)
                t_stt_end = time.perf_counter()

                if not text:
                    print("  (sin voz reconocible)")
                    continue
                print(f"  Tú: {text}")

                ttft, total_llm = _timed_chat(text)
                total = (t_stt_end - t_stt_start) + total_llm
                print(
                    f"  [tiempos] STT: {t_stt_end - t_stt_start:.1f}s | "
                    f"primer token LLM: {ttft:.2f}s | "
                    f"total (STT+LLM): {total:.1f}s"
                )
            except KeyboardInterrupt:
                print("\nSalida.")
                break
            except Exception as e:
                print(f"  (error: {e}) — reintentando...")
                time.sleep(1.0)
    finally:
        stt_close()

if __name__ == "__main__":
    main()
