import re
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ROOT
from .summarize import notes_to_markdown, summarize
from .transcript import merge_segments, to_text


def load_audio(path):
    """16 kHz mono 16-bit WAVs load directly; anything else is handed to faster-whisper (PyAV)."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        with wave.open(str(p), "rb") as w:
            if w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2:
                return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return str(p)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "meeting"


def run_pipeline(tracks: dict, title: str, cfg, started: datetime = None) -> Path:
    """`tracks` maps a speaker label ('You', 'Others', ...) to audio (int16 array or file path)."""
    from .transcribe import Transcriber

    started = started or datetime.now()
    asr = Transcriber(cfg)
    per_track, duration = [], 0.0
    t0 = time.perf_counter()
    for speaker, audio in tracks.items():
        if isinstance(audio, np.ndarray) and (len(audio) == 0 or np.abs(audio).max() < 30):
            print(f"  {speaker}: track is silent, skipping")
            continue
        segs, dur = asr.transcribe_track(audio, speaker)
        per_track.append(segs)
        duration = max(duration, dur)
    asr_s = time.perf_counter() - t0
    if duration:
        print(f"Transcribed {duration:.0f}s of audio in {asr_s:.0f}s (RTF {asr_s / duration:.2f})")

    transcript = to_text(merge_segments(per_track))
    if not transcript.strip():
        raise SystemExit("No speech found in the recording.")

    notes = None
    try:
        print("Summarising ...")
        notes = summarize(transcript, cfg)
    except Exception as exc:
        print(f"Summary failed ({exc}); saving transcript only.")

    out_dir = ROOT / cfg.notes_dir
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{started:%Y-%m-%d_%H%M}-{slug(title)}.md"
    path.write_text(
        notes_to_markdown(title, f"{started:%d %b %Y, %H:%M}", duration, notes, transcript),
        encoding="utf-8",
    )
    return path
