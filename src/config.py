"""Configuracion del Language Orchestrator.
"""

import os
import sys
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Carga de .env (raiz del repo)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Carga el ``.env`` de la raiz del repo."""
    
    env_file = Path(__file__).resolve().parent.parent / ".env"

    if not env_file.is_file():
        return

    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

    except OSError as e:
        print(f"[config] no se pudo leer .env: {e}", file=sys.stderr)

_load_dotenv()

class ConfigError(RuntimeError):
    """Configuracion incompleta o invalida (falta una clave de entorno)."""

class Settings:
    """Acceso de solo lectura a la configuracion, por nombre de clave.
    """

    REQUIRED: ClassVar[tuple[str, ...]] = (
        # LLM - Docker Model Runner (OpenAI-compatible)
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",

        # Prompt / historial
        "SYSTEM_PROMPT",
        "HISTORY_TURNS",

        # STT - ScribeCrate Live
        "STT_TRANSPORT",
        "STT_WS_URL",
        "STT_REST_URL",
        "STT_MODEL",
        "STT_LANG",
        "STT_USE_VAD",

        # Microfono / corte de frase
        "SILENCE_THRESHOLD",
        "SILENCE_SECONDS",
        "MAX_RECORD_SECONDS",

        # Flujo
        "VERBOSE",
    )

    OPTIONAL: ClassVar[tuple[str, ...]] = (
        "LLM_API_KEY",
        "LLM_EXTRA_BODY",
        "STT_API_KEY",
        "AUDIO_INPUT_DEVICE",
        "SILENCE_RECALIBRATE_EVERY",
    )

    _KNOWN: ClassVar[set[str]] = set(REQUIRED) | set(OPTIONAL)

    def __init__(self) -> None:
        missing = [
            k for k in self.REQUIRED if not (os.environ.get(k) or "").strip()
        ]
        if missing:
            raise ConfigError(
                "Configuracion incompleta: faltan las siguientes variables "
                "de entorno (o en el .env de la raiz del repo):\n  "
                + "\n  ".join(missing)
                + "\nVer .env.example para la lista completa de claves."
            )

    # ------------------------------------------------------------------
    # Acceso por nombre de clave
    # ------------------------------------------------------------------

    def get(self, name: str) -> str | None:
        """Devuelve el valor de la clave ``name``.

        - Clave requerida: string no vacio (siempre).
        - Clave opcional: string, o ``None`` si no esta definida.
        - Clave desconocida: lanza :class:`ConfigError`.
        """
        if name in self.REQUIRED:
            return os.environ[name]
        if name in self.OPTIONAL:
            value = os.environ.get(name)
            return value if (value or "").strip() else None
        raise ConfigError(f"Clave de configuracion desconocida: {name!r}")

    # ------------------------------------------------------------------
    # Accessors tipados (sin defaults)
    # ------------------------------------------------------------------

    @property
    def llm_temperature(self) -> float:
        return float(self.get("LLM_TEMPERATURE"))

    @property
    def history_turns(self) -> int:
        return int(self.get("HISTORY_TURNS"))

    @property
    def silence_threshold(self) -> float:
        return float(self.get("SILENCE_THRESHOLD"))

    @property
    def silence_recibrate_every(self) -> int | None:
        """Frases entre recalibraciones del umbral (None/0 = una sola vez)."""
        value = self.get("SILENCE_RECALIBRATE_EVERY")
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @property
    def silence_seconds(self) -> float:
        return float(self.get("SILENCE_SECONDS"))

    @property
    def max_record_seconds(self) -> float:
        return float(self.get("MAX_RECORD_SECONDS"))

    @property
    def stt_use_vad(self) -> bool:
        return self.get("STT_USE_VAD").strip().lower() in ("1", "true", "yes", "on")

    @property
    def stt_lang(self) -> str | None:
        """Codigo de idioma del STT; ``None`` si es autodeteccion ("auto"/vacio)."""
        raw = (self.get("STT_LANG") or "").strip()
        return None if raw.lower() in ("", "auto") else raw

    @property
    def verbose(self) -> bool:
        return self.get("VERBOSE").strip().lower() not in ("0", "false", "no", "off")

    @property
    def audio_input_device(self) -> int | None:
        value = self.get("AUDIO_INPUT_DEVICE")
        return int(value) if value is not None else None

settings = Settings()
# ---------------------------------------------------------------------------
# Constantes de audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
CHANNELS: int = 1
DTYPE: str = "float32"
BLANK_AUDIO_THRESHOLD: float = 0.0001

# ---------------------------------------------------------------------------
# Conveniencia: acceso por nombre de clave a nivel de modulo (PEP 562)
# ---------------------------------------------------------------------------
def __getattr__(name: str):
    if name in Settings._KNOWN:
        return settings.get(name)

    raise AttributeError(f"module 'src.config' has no attribute {name!r}")

__all__ = [
    "Settings",
    "ConfigError",
    "settings",
    "SAMPLE_RATE",
    "CHANNELS",
    "DTYPE",
    "BLANK_AUDIO_THRESHOLD",
]

# ---------------------------------------------------------------------------
# Compatibilidad con la consola Windows
# ---------------------------------------------------------------------------
if sys.platform.startswith("win"):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
