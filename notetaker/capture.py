"""Audio capture: microphone via sounddevice, system audio via WASAPI loopback (soundcard)."""
import threading
import warnings
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono

warnings.filterwarnings("ignore", message="data discontinuity")


def resolve_input_device(spec):
    """Turn '' / '2' / 'Microphone Array' into a sounddevice input index (or None = default)."""
    if spec in (None, ""):
        return None
    if str(spec).isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower():
            return i
    raise ValueError(f"No input device matching '{spec}'. Run: python -m notetaker devices")


def _concat(chunks):
    chunks = list(chunks)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def save_wav(path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())


class MicTrack:
    def __init__(self, device=""):
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=self._callback,
            device=resolve_input_device(device),
        )

    def _callback(self, indata, frames, time_info, status):
        self._chunks.append(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()
        return _concat(self._chunks)


class LoopbackTrack:
    """Records whatever plays through the speakers (the remote side of a call)."""

    def __init__(self, device=""):
        import soundcard as sc

        speakers = sc.all_speakers()
        speaker = sc.default_speaker()
        if device:
            matches = [s for s in speakers if str(device).lower() in s.name.lower()]
            if not matches:
                raise ValueError(f"No speaker matching '{device}'. Run: python -m notetaker devices")
            speaker = matches[0]
        self.speaker_name = speaker.name
        self._mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self._chunks = []
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self.error = None

    def _run(self):
        try:
            channels = max(1, min(2, self._mic.channels))
            with self._mic.recorder(samplerate=SAMPLE_RATE, channels=channels) as rec:
                self._ready.set()
                while not self._stop.is_set():
                    data = rec.record(numframes=SAMPLE_RATE // 4)  # 0.25 s blocks
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    self._chunks.append((np.clip(mono, -1, 1) * 32767).astype(np.int16))
        except Exception as exc:  # reported to the caller via .error
            self.error = exc
            self._ready.set()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self.error:
            raise RuntimeError(self.error)

    def stop(self):
        self._stop.set()
        # WASAPI loopback can block while nothing is playing, so never wait forever.
        self._thread.join(timeout=3)
        return _concat(self._chunks)
