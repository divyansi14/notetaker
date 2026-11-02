import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
from faster_whisper import WhisperModel

from .transcript import Segment, drop_hallucinations


def peak_level(audio) -> float:
    """Peak amplitude (0..1) of an int16 array; 0.0 for anything else."""
    if isinstance(audio, np.ndarray) and audio.dtype == np.int16 and len(audio):
        return float(np.abs(audio.astype(np.int32)).max()) / 32768.0
    return 0.0


def to_float(audio):
    """int16 -> float32 in [-1, 1]; quiet tracks are boosted (max 8x) so Whisper hears them."""
    if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
        x = audio.astype(np.float32) / 32768.0
        peak = peak_level(audio)
        if 0.001 < peak < 0.5:
            x *= min(0.7 / peak, 8.0)
        return x
    return audio


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        vocab = [w for w in cfg.vocabulary if str(w).strip()]
        self.prompt = ("Vocabulary: " + ", ".join(vocab) + ".") if vocab else None
        print(f"Loading Whisper model '{cfg.model}' ...")
        self.model = WhisperModel(
            cfg.model,
            device="cpu",
            compute_type=cfg.compute_type,
            cpu_threads=cfg.cpu_threads,
        )

    def transcribe_track(self, audio, speaker: str):
        """Returns (segments, audio_seconds). `audio` is an int16/float32 array or a file path."""
        peak = peak_level(audio)
        if peak:
            print(f"  {speaker}: input level (peak) {peak:.2f}")
        segments, info = self.model.transcribe(
            to_float(audio),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            initial_prompt=self.prompt,
        )
        total = info.duration
        out = []
        for s in segments:  # generator: transcription happens as we iterate
            text = s.text.strip()
            if text:
                out.append(Segment(s.start, s.end, speaker, text))
            print(f"\r  {speaker}: transcribed {s.end:6.0f}s / {total:.0f}s", end="", flush=True)
        print()
        return drop_hallucinations(out), total