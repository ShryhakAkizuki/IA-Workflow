"""
Cliente STT para ScribeCrate Live (hwdsl2/whisper-live-server).

Soporta dos transportes, seleccionables con la variable STT_TRANSPORT:

    1. "websocket" (por defecto, el preferido):
    2. "rest" (fallback):

El audio se lee desde archivos WAV (16 kHz mono float32 o PCM 16-bit)
usando la funcion local _load_wav, sin dependencias externas.
"""

import json
import os
import struct
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

def _transcribe_ws(pcm_f32: bytes) -> Dict[str, Any]:
    """
    Transcribe stream de audio float32 16 kHz via WebSocket de ScribeCrate.
    """
    import websocket

    url = _build_ws_url()

    if config.VERBOSE:
        host = urllib.parse.urlparse(url).netloc
        print(f"[STT:WS] conectando a ws://{host} ...")

    t0 = time.monotonic()
    ws = websocket.create_connection(url, timeout=30)

    segments: Dict[float, str] = {}
    translations: List[str] = []
    try:

        lang = None if config.STT_LANG.lower() in ("auto", "") else config.STT_LANG
        cfg: Dict[str, Any] = {
            "uid": "language-orchestrator",
            "language": lang,
            "model": config.STT_MODEL or None,
            "task": "transcribe",
            "audio_format": "float32",
            "use_vad": config.STT_USE_VAD in ("1", "true", "yes"),
        }
        ws.send(json.dumps(cfg))
        if config.VERBOSE:
            print(
                f"[STT:WS] config enviada: model={cfg['model']}, "
                f"lang={cfg['language']}, vad={cfg['use_vad']}"
            )

        ws.settimeout(30)
        while True:
            msg = _parse_ws_msg(ws.recv())
            if _msg_is_ready(msg):
                break
            _collect_segments(msg, segments, translations)
        if config.VERBOSE:
            print(f"[STT:WS] SERVER_READY en {time.monotonic() - t0:.1f}s")

        CHUNK = 32
        n_chunks = (len(pcm_f32) + CHUNK - 1) // CHUNK

        for i in range(n_chunks):
            ws.send_binary(pcm_f32[i * CHUNK : (i + 1) * CHUNK])

        if config.VERBOSE:
            secs = len(pcm_f32) / 4 / config.SAMPLE_RATE
            print(
                f"[STT:WS] audio enviado: {n_chunks} chunks "
                f"({secs:.1f}s de audio) en {time.monotonic() - t0:.1f}s"
            )

        tail = b"\x00\x00\x00\x00" * (config.SAMPLE_RATE // 2)
        for i in range(0, len(tail), CHUNK):
            ws.send_binary(tail[i : i + CHUNK])

        ws.settimeout(2.0)
        quiet_since = time.monotonic()
        deadline = time.monotonic() + 90.0

        while time.monotonic() < deadline:
            try:
                msg = _parse_ws_msg(ws.recv())

            except websocket.WebSocketTimeoutException:
                if segments and time.monotonic() - quiet_since > 3.0:
                    break
                continue

            except websocket.WebSocketException:
                break  # el servidor cerro la conexion
            quiet_since = time.monotonic()

            if _msg_is_ready(msg):
                continue
            _collect_segments(msg, segments, translations)

        total = time.monotonic() - t0
        if config.VERBOSE:
            print(f"[STT:WS] {len(segments)} segmentos en {total:.1f}s")

        return {
            "text": "".join(segments[k] for k in sorted(segments)).strip(),
            "translations": translations,
            "segments": [
                {"start": k, "text": segments[k]} for k in sorted(segments)
            ],
            "transport": "websocket",
        }

    finally:
        try:
            ws.close()
        except Exception:
            pass

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

    lang = None if config.STT_LANG.lower() in ("auto", "") else config.STT_LANG

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
