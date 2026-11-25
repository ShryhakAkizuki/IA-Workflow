import numpy as np
import sounddevice as sd

from src.config import SAMPLE_RATE, settings

_BLOCK_SAMPLES = 480
_FRAME_SECONDS = _BLOCK_SAMPLES / SAMPLE_RATE
MAX_IDLE_SECONDS = 15.0

def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))

def _input_device():
    override = settings.audio_input_device
    if override is not None:
        return override
    return sd.default.device[0]

def _adaptive_threshold() -> float:

    base = settings.silence_threshold
    try:
        device = _input_device()
        n_blocks = max(2, int(0.6 / _FRAME_SECONDS))
        energies: list[float] = []
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_BLOCK_SAMPLES,
            device=device,
        ) as stream:
            for _ in range(n_blocks):
                data, _ = stream.read(_BLOCK_SAMPLES)
                energies.append(_rms(data[:, 0]))

        floor = float(np.percentile(energies, 25))
        peak = float(np.max(energies))

        if peak > max(floor * 2.0, 1e-6):
            floor = floor / 2.0
        return max(base, min(floor * 4.0, 0.2))
    except Exception:
        return base

def record_until_silence() -> np.ndarray:

    threshold = _adaptive_threshold()
    voiced: list[np.ndarray] = []
    started = False
    silence_acc = 0.0
    idle_acc = 0.0
    device = _input_device()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=_BLOCK_SAMPLES,
        device=device,
    ) as stream:
        while True:
            indata, _overflowed = stream.read(_BLOCK_SAMPLES)
            block = indata[:, 0]
            energy = _rms(block)

            if not started:
                if energy > threshold:
                    started = True
                else:
                    idle_acc += _FRAME_SECONDS
                    if idle_acc >= MAX_IDLE_SECONDS:
                        return np.empty(0, dtype=np.float32)
                continue

            voiced.append(block)
            if energy > threshold:
                silence_acc = 0.0
            else:
                silence_acc += _FRAME_SECONDS

            total_seconds = sum(len(b) for b in voiced) / SAMPLE_RATE
            if silence_acc >= settings.silence_seconds or total_seconds >= settings.max_record_seconds:
                break

    if not voiced:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(voiced)
