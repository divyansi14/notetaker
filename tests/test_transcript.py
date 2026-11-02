import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notetaker.transcript import Segment, drop_hallucinations, fmt_time, merge_segments, to_text  # noqa: E402


def test_fmt_time():
    assert fmt_time(65) == "01:05"
    assert fmt_time(3725) == "1:02:05"


def test_merge_interleaves_tracks_and_joins_same_speaker():
    you = [Segment(0, 3, "You", "Hello everyone."), Segment(4, 6, "You", "Let's begin.")]
    others = [Segment(10, 12, "Others", "Sounds good.")]
    merged = merge_segments([you, others])
    assert [s.speaker for s in merged] == ["You", "Others"]
    assert merged[0].text == "Hello everyone. Let's begin."
    assert merged[0].end == 6


def test_merge_does_not_join_across_large_gap():
    you = [Segment(0, 3, "You", "First."), Segment(20, 22, "You", "Second.")]
    assert len(merge_segments([you])) == 2


def test_drop_hallucinations_and_loops():
    segs = [Segment(0, 1, "You", "Thank you."), Segment(2, 5, "You", "Real content here.")]
    assert [s.text for s in drop_hallucinations(segs)] == ["Real content here."]
    loop = [Segment(i, i + 3, "You", "same line") for i in range(0, 12, 3)]
    assert len(drop_hallucinations(loop)) == 2


def test_to_text_format():
    text = to_text([Segment(65, 70, "Others", "Ship it.")])
    assert text == "[01:05] Others: Ship it."
