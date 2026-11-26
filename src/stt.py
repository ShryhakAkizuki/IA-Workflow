"""
Cliente STT para ScribeCrate Live (hwdsl2/whisper-live-server).

Soporta dos transportes, seleccionables con la variable STT_TRANSPORT:

    1. "websocket" (por defecto, el preferido)
    2. "rest" (fallback)

El audio se lee desde archivos WAV (16 kHz mono float32 o PCM 16-bit)
usando la funcion local _load_wav, sin dependencias externas.
"""

import json
import os
import struct
import threading
import time
import urllib.parse
import wave
from typing import Any, Dict, List, Optional

from src import config

# ---------------------------------------------------------------------------
# Carga de audio local
# ---------------------------------------------------------------------------
def _load_wav(ws_path: str) -> bytes:
    """
    Lee un archivo WAV y devuelve el PCM float32 mono 16 kHz (little-endian).

    Acepta WAV PCM de 16 o 32 bits (int o float) en cualquier canal/tasa:
    convierte a mono y remuestrea linealmente a 16 kHz.
    """
    with wave.open(ws_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    n = n_frames * n_channels

    if sampwidth == 2:
        samples = [s / 32768.0 for s in struct.unpack(f"<{n}h", raw[: n * 2])]

    elif sampwidth == 4:
        samples = list(struct.unpack(f"<{n}f", raw[: n * 4]))

    else:
        raise ValueError(
            f"ScribeCrate STT: anchura de muestra no soportada: {sampwidth * 8} bits"
        )

    if n_channels > 1:
        mono = [
            sum(samples[i * n_channels : (i + 1) * n_channels]) / n_channels
            for i in range(n_frames)
        ]
    
    else:
        mono = samples

    if framerate != config.SAMPLE_RATE:
        ratio = config.SAMPLE_RATE / framerate
        n_out = int(len(mono) * ratio)
        out = [0.0] * n_out

        for i in range(n_out):
            pos = i / ratio
            lo = int(pos)
            hi = min(lo + 1, len(mono) - 1)
            frac = pos - lo
            out[i] = mono[lo] * (1.0 - frac) + mono[hi] * frac

        mono = out

    return struct.pack(f"<{len(mono)}f", *mono)

# ---------------------------------------------------------------------------
# Transporte WebSocket (tiempo real)
# ---------------------------------------------------------------------------
def _build_ws_url() -> str:
    """Arma la URL WS anadiendo el token si hay API key y no esta presente."""

    url = config.STT_WS_URL.rstrip()

    if not config.STT_API_KEY:
        return url

    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    if "token" in params or "Authorization" in params:
        return url

    sep = "&" if parsed.query else "?"

    return f"{url}{sep}token={urllib.parse.quote(config.STT_API_KEY)}"

def _parse_ws_msg(raw: Any) -> Any:
    """Decodifica un mensaje WS; devuelve un dict si es JSON, None si no."""

    if not isinstance(raw, str):
        return None

    try:
        parsed = json.loads(raw)

    except (ValueError, TypeError):
        return None

    return parsed if isinstance(parsed, dict) else None

def _msg_is_ready(msg: Any) -> bool:
    return isinstance(msg, dict) and msg.get("message") == "SERVER_READY"

def _merge_segment(
    s: Dict[str, Any], segments: Dict[float, str], translations: List[str]
) -> None:
    """Combina un segmento (los eventos son acumulativos; gana el texto mas largo
    o el marcado como completed) en el dict clave=start."""

    if msg_type := s.get("type"):
        if msg_type == "translation" and "text" in s:
            translations.append(str(s["text"]))
            return

    try:
        key = round(float(s.get("start", 0.0)), 2)

    except (TypeError, ValueError):
        key = 0.0

    text = str(s.get("text", ""))
    prev = segments.get(key)

    if s.get("completed") or prev is None or len(text) > len(prev):
        segments[key] = text

def _collect_segments(
    msg: Any, segments: Dict[float, str], translations: List[str]
) -> None:
    """Extrae segmentos de un mensaje del servidor (formato actual y legado)."""

    if not isinstance(msg, dict):
        return

    events = msg.get("segments")

    if isinstance(events, list):
        for s in events:
            if isinstance(s, dict) and "text" in s:
                _merge_segment(s, segments, translations)

        return

    if isinstance(msg.get("segment"), dict):
        _merge_segment(msg["segment"], segments, translations)
        return

    if msg.get("type") == "translation" and "text" in msg:
        translations.append(str(msg["text"]))

# --- Canal persistente -----
_WS_CHUNK = 16 * 1024
_WS_READY_TIMEOUT = 30.0
_WS_DRAIN_QUIET = 2.0
_WS_DRAIN_DEADLINE = 60.0

def _msg_all_completed(msg: Any) -> bool:
    """
    True si el mensaje trae la lista de segmentos y TODOS llegan con
    "completed": el servidor termino de procesar la frase (los eventos son
    acumulativos, asi que el ultimo mensaje con todo "completed" es el final).
    """
    if not isinstance(msg, dict):
        return False

    events = msg.get("segments")

    if not isinstance(events, list) or not events:
        return False

    return all(isinstance(s, dict) and s.get("completed") for s in events)

_ws_standby: Optional[Any] = None
_ws_lock = threading.Lock()
_ws_regen_thread: Optional[threading.Thread] = None


def _ws_config() -> Dict[str, Any]:
    """Payload de config para ScribeCrate."""
    return {
        "uid": "language-orchestrator",
        "language": config.settings.stt_lang,
        "model": config.STT_MODEL or None,
        "task": "transcribe",
        "audio_format": "float32",
        "use_vad": config.settings.stt_use_vad,
    }


def _ws_open_session() -> Any:
    """Abre una conexion WebSocket nueva y espera su SERVER_READY."""
    import websocket

    url = _build_ws_url()

    if config.VERBOSE:
        host = urllib.parse.urlparse(url).netloc
        print(f"[STT:WS] abriendo sesion (ws://{host}) ...")

    t0 = time.monotonic()
    ws = websocket.create_connection(url, timeout=30)
    ws.send(json.dumps(_ws_config()))
    ws.settimeout(_WS_READY_TIMEOUT)

    while True:
        msg = _parse_ws_msg(ws.recv())

        if _msg_is_ready(msg):
            break

    if config.VERBOSE:
        print(f"[STT:WS] sesion lista en {time.monotonic() - t0:.2f}s")

    return ws


def _ws_regen() -> None:
    """Abre una nueva sesion en segundo plano (para la proxima frase)."""
    global _ws_regen_thread

    with _ws_lock:

        if _ws_standby is not None:
            return  # ya hay sesion en standby

        if _ws_regen_thread is not None and _ws_regen_thread.is_alive():
            return  # ya hay una regeneracion en marcha

        _ws_regen_thread = threading.Thread(target=_ws_regen_worker, daemon=True)
        _ws_regen_thread.start()


def _ws_regen_worker() -> None:
    global _ws_standby

    last_err: Optional[Exception] = None
    for attempt, delay in ((1, 0.0), (2, 1.0), (3, 2.0), (4, 4.0)):
        with _ws_lock:
            if _ws_standby is not None:
                return
        try:
            time.sleep(delay)
            ws = _ws_open_session()
        except Exception as e:
            last_err = e
            continue
        with _ws_lock:
            if _ws_standby is None:
                _ws_standby = ws
                if config.VERBOSE:
                    print("[STT:WS] nueva sesion en standby")
        return
    print(
        f"[STT:WS] regeneracion en segundo plano fallo tras 4 intentos ({last_err}); "
        "se reconectara en la proxima frase"
    )


def _ws_take_session() -> Any:
    """Consume la sesion standby (o abre una de forma sincrona como fallback)."""
    global _ws_standby

    with _ws_lock:
        ws = _ws_standby
        _ws_standby = None
    if ws is not None:
        return ws
    ws = _ws_open_session()
    _ws_regen()
    return ws


def _ws_discard(ws: Any) -> None:
    """Cierra una sesion consumida (el servidor la cerraria por inactividad)."""
    try:
        ws.close()
    except Exception:
        pass


def _ws_drain(ws: Any) -> tuple[Dict[float, str], List[str]]:
    """
    Lee segmentos hasta que TODOS los segmentos lleguen "completed" y
    sigan ``_WS_DRAIN_QUIET`` s sin mas mensajes (o deadline). El servidor
    deja de enviar nada una vez la frase se completa.
    """
    import websocket

    segments: Dict[float, str] = {}
    translations: List[str] = []
    ws.settimeout(2.0)
    quiet_since = time.monotonic()
    all_completed_at: Optional[float] = None
    deadline = time.monotonic() + _WS_DRAIN_DEADLINE

    while time.monotonic() < deadline:
        try:
            msg = _parse_ws_msg(ws.recv())

        except websocket.WebSocketTimeoutException:

            if all_completed_at is not None and time.monotonic() - all_completed_at > _WS_DRAIN_QUIET:
                break
            continue

        except websocket.WebSocketException:
            break

        quiet_since = time.monotonic()

        if _msg_is_ready(msg):
            continue
        _collect_segments(msg, segments, translations)

        if _msg_all_completed(msg):
            all_completed_at = max(all_completed_at or quiet_since, quiet_since)

    return segments, translations


def _transcribe_ws(pcm_f32: bytes) -> Dict[str, Any]:
    """
    Transcribe un stream de audio float32 16 kHz sobre una sesion ya lista
    en SERVER_READY (standby). No se repite el handshake ni la espera de
    ready por frase: mientras transcibe, se regenera en segundo plano la
    sesion de la proxima frase.
    """
    ws = _ws_take_session()
    t0 = time.monotonic()
    try:
        n_chunks = (len(pcm_f32) + _WS_CHUNK - 1) // _WS_CHUNK
        for i in range(n_chunks):
            ws.send_binary(pcm_f32[i * _WS_CHUNK : (i + 1) * _WS_CHUNK])

        if config.VERBOSE:
            secs = len(pcm_f32) / 4 / config.SAMPLE_RATE
            print(
                f"[STT:WS] audio enviado: {n_chunks} chunks "
                f"({secs:.1f}s de audio) en {time.monotonic() - t0:.1f}s"
            )

        tail = b"\x00\x00\x00\x00" * (config.SAMPLE_RATE // 2)
        for i in range(0, len(tail), _WS_CHUNK):
            ws.send_binary(tail[i : i + _WS_CHUNK])

        segments, translations = _ws_drain(ws)

        total = time.monotonic() - t0
        if config.VERBOSE:
            print(
                f"[STT:WS] {len(segments)} segmentos en {total:.1f}s "
                "(sin handshake ni espera de ready)"
            )

        return {
            "text": "".join(segments[k] for k in sorted(segments)).strip(),
            "translations": translations,
            "segments": [
                {"start": k, "text": segments[k]} for k in sorted(segments)
            ],
            "transport": "websocket",
        }
    finally:
        _ws_discard(ws)
        _ws_regen()


def stt_warmup() -> None:
    """
    Abre una sesion standby y carga el modelo para que la primera frase no
    espere el handshake ni el SERVER_READY.
    """
    if config.STT_TRANSPORT == "rest":
        return
    _ws_regen()


def stt_close() -> None:
    """Cierra la sesion standby (y deja de regenerar nuevas sesions)."""
    global _ws_standby

    with _ws_lock:
        ws = _ws_standby
        _ws_standby = None
    if ws is not None:
        _ws_discard(ws)
        if config.VERBOSE:
            print("[STT:WS] sesion cerrada")

# ---------------------------------------------------------------------------
# Transporte REST (fallback, OpenAI-compatible)
# ---------------------------------------------------------------------------
def _transcribe_rest(pcm_f32: bytes, wav_path: str) -> Dict[str, Any]:
    """
    Transcribe el archivo completo via REST OpenAI-compatible de ScribeCrate.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=config.STT_REST_URL.rstrip("/") + "/v1",
        api_key=config.STT_API_KEY or "no-key",
        timeout=300.0,
    )

    lang = config.settings.stt_lang

    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=f,
            model=config.STT_MODEL or "large-v3-turbo",
            language=lang,
        )

    return {
        "text": (getattr(resp, "text", "") or "").strip(),
        "translations": [],
        "segments": [],
        "transport": "rest",
    }

# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def transcribe_audio(audio: Any) -> str:
    """
    Transcribe audio float32 mono 16 kHz en memoria (numpy o lista)..
    """
    import numpy as np

    arr = np.ascontiguousarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    pcm_f32 = arr.tobytes()

    if len(pcm_f32) == 0:
        return ""

    if config.STT_TRANSPORT == "rest":
        import tempfile
        import wave as _wave

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=os.environ.get("TEMP", ".")
        ) as tmp:
            tmp_path = tmp.name
        try:
            with _wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(pcm_f32)
            result = _transcribe_rest(pcm_f32, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        result = _transcribe_ws(pcm_f32)

    return result["text"]

def transcribe_wav(ws_path: str) -> str:
    """
    Transcribe un archivo WAV local usando el transporte configurado
    Devuelve el texto de la transcripcion.
    """

    if not os.path.isfile(ws_path):
        raise FileNotFoundError(f"ScribeCrate STT: no existe el archivo: {ws_path}")

    pcm_f32 = _load_wav(ws_path)
    secs = len(pcm_f32) / 4 / config.SAMPLE_RATE

    if config.VERBOSE:
        print(f"[STT] cargando {os.path.basename(ws_path)} ({secs:.1f}s @ 16 kHz mono)")

    if config.STT_TRANSPORT == "rest":
        result = _transcribe_rest(pcm_f32, ws_path)

    else:
        result = _transcribe_ws(pcm_f32)

    return result["text"]
