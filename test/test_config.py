"""Prueba de seguridad del cargador de configuracion.

Uso:
    python test/test_config.py
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = [
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_EXTRA_BODY",
    "LLM_TEMPERATURE",
    "SYSTEM_PROMPT",
    "HISTORY_TURNS",
    "STT_TRANSPORT",
    "STT_WS_URL",
    "STT_REST_URL",
    "STT_MODEL",
    "STT_LANG",
    "STT_USE_VAD",
    "SILENCE_THRESHOLD",
    "SILENCE_SECONDS",
    "MAX_RECORD_SECONDS",
    "VERBOSE",
]


def _purge_env() -> None:
    for key in REQUIRED + ["LLM_API_KEY", "STT_API_KEY", "AUDIO_INPUT_DEVICE"]:
        os.environ.pop(key, None)
    env_file = Path(__file__).resolve().parents[1] / ".env"

    if env_file.is_file():
        env_file.unlink()


def main() -> int:
    env_backup = None
    env_path = Path(__file__).resolve().parents[1] / ".env"

    if env_path.is_file():
        env_backup = env_path.read_bytes()
        env_path.unlink()

    try:
        _purge_env()
        sys.modules.pop("src.config", None)
        sys.modules.pop("src", None)

        try:
            importlib.import_module("src.config")

        except RuntimeError as e:
            missing_in_msg = [k for k in REQUIRED if k in str(e)]
            assert len(missing_in_msg) == len(REQUIRED), (
                f"El error no lista todas las variables faltantes: {e!r}"
            )
            print("[OK] src.config rechazo el arranque sin variables de entorno.")
            return 0

        print("[FALLO] src.config se importo sin variables de entorno (usaria defaults).")
        return 1
    finally:

        for mod in ("src.config", "src"):
            sys.modules.pop(mod, None)
        if env_backup is not None:
            env_path.write_bytes(env_backup)
            print(f"[OK] .env restaurado: {env_path}")


if __name__ == "__main__":
    raise SystemExit(main())
