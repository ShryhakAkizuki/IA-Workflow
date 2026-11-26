"""Prueba de seguridad del cargador de configuracion.

Verifica:
  1. Que ``src.config`` rechaza el arranque si faltan las variables
     REQUERIDAS, listandolas todas en el error.
  2. Que ``STT_USE_VAD=on`` y ``STT_USE_VAD=1`` producen el mismo payload
     WS (properties tipadas, item 0.2 de PLAN_MEJORAS.md).

El ``.env`` real se oculta durante la prueba (renombrandolo) y siempre se
restaura; nunca se borra.

Uso:
    python test/test_config.py

Precondicion: el ``.env`` de la raiz debe ser valido (se necesita una unica
vez al importar ``src.config`` para leer ``Settings.REQUIRED``).
"""
import contextlib
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # se importa con el .env real (debe ser valido)

REQUIRED = tuple(Settings.REQUIRED)
ALL_KEYS = tuple(set(REQUIRED) | set(Settings.OPTIONAL))
_MODULES = ("src.stt", "src.config", "src")


def _purge_env() -> None:
    for key in ALL_KEYS:
        os.environ.pop(key, None)


def _populate_env(**overrides: str) -> None:
    """Deja un entorno minimo: solo las REQUERIDAS (valor dummy) + overrides."""
    _purge_env()
    for key in REQUIRED:
        os.environ[key] = "dummy"
    for key, value in overrides.items():
        os.environ[key] = value


def _reload_config():
    for mod in _MODULES:
        sys.modules.pop(mod, None)
    return importlib.import_module("src.config")


def _drop_modules() -> None:
    for mod in _MODULES:
        sys.modules.pop(mod, None)


@contextlib.contextmanager
def _env_hidden():
    """Oculta el .env real (renombrandolo) para que el loader no lo lea."""
    env_path = ROOT / ".env"
    hidden = ROOT / ".env.test_config_backup"
    if hidden.exists():
        hidden.unlink()  # residuo de una corrida anterior interrumpida
    moved = env_path.is_file()
    if moved:
        env_path.rename(hidden)
    try:
        yield
    finally:
        if moved and hidden.exists():
            hidden.rename(env_path)


def test_missing_required() -> bool:
    """Sin variables de entorno, el import debe fallar y listar todas."""
    _purge_env()
    try:
        try:
            _reload_config()
        except RuntimeError as e:
            missing_in_msg = [k for k in REQUIRED if k in str(e)]
            if len(missing_in_msg) != len(REQUIRED):
                print(f"[FALLO] El error no lista todas las variables faltantes: {e!r}")
                return False
            print("[OK] src.config rechazo el arranque sin variables de entorno.")
            return True

        print("[FALLO] src.config se importo sin variables de entorno (usaria defaults).")
        return False
    finally:
        _drop_modules()


def test_vad_alias_on_equals_one() -> bool:
    """STT_USE_VAD=on y =1 deben producir el mismo payload WS (use_vad=True)."""
    try:
        _populate_env(STT_USE_VAD="on")
        _reload_config()
        stt = importlib.import_module("src.stt")
        payload_on = stt._ws_config()

        _populate_env(STT_USE_VAD="1")
        _reload_config()
        stt = importlib.import_module("src.stt")
        payload_one = stt._ws_config()

        if payload_on.get("use_vad") is not True:
            print(f"[FALLO] con STT_USE_VAD=on se esperaba use_vad=True: {payload_on!r}")
            return False
        if payload_on != payload_one:
            print(f"[FALLO] los payloads difieren: on={payload_on!r} 1={payload_one!r}")
            return False

        print("[OK] STT_USE_VAD=on y STT_USE_VAD=1 producen el mismo payload WS.")
        return True
    finally:
        _drop_modules()


def main() -> int:
    snapshot = {k: os.environ[k] for k in ALL_KEYS if k in os.environ}
    try:
        with _env_hidden():
            results = [test_missing_required(), test_vad_alias_on_equals_one()]
    finally:
        _purge_env()
        os.environ.update(snapshot)
        _drop_modules()

    print()
    if all(results):
        print("[OK] test_config: todas las pruebas pasaron (.env intacto).")
        return 0

    print("[FALLO] test_config: alguna prueba fallo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
