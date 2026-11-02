"""Pure transcript logic (no audio or model imports): segments, merging, formatting."""
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


HALLUCINATIONS = {"thank you.", "thanks for watching.", "thank you for watching.", "you", "bye."}


def drop_hallucinations(segs):
    """Remove stock phantom phrases on short segments and runaway repeated lines."""
    out, last, repeat = [], None, 0
    for s in segs:
        t = s.text.strip().lower()
        if (s.end - s.start) < 2.5 and t in HALLUCINATIONS:
            continue
        if t == last:
            repeat += 1
        else:
            last, repeat = t, 0
        if repeat >= 2:  # third identical line in a row is a decoding loop
            continue
        out.append(s)
    return out


def merge_segments(tracks, gap: float = 2.0):
    """Merge per-track segment lists into one timeline; join same-speaker lines within `gap` s."""
    flat = sorted((s for track in tracks for s in track), key=lambda s: (s.start, s.end))
    merged = []
    for s in flat:
        if merged and merged[-1].speaker == s.speaker and s.start - merged[-1].end <= gap:
            prev = merged[-1]
            merged[-1] = Segment(prev.start, max(prev.end, s.end), prev.speaker, f"{prev.text} {s.text}")
        else:
            merged.append(Segment(s.start, s.end, s.speaker, s.text))
    return merged


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def to_text(segs) -> str:
    return "\n".join(f"[{fmt_time(s.start)}] {s.speaker}: {s.text}" for s in segs)
