import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notetaker.summarize import Notes, chunk_text, notes_to_markdown, parse_notes_json  # noqa: E402

RAW = """```json
{"summary": "We planned the launch.", "key_points": ["Launch Monday"],
 "decisions": ["Ship on Monday"],
 "action_items": [{"owner": "Asha", "task": "Send notes", "due": "Friday"}, "Book the room", {"task": ""}]}
```"""


def test_parse_fenced_json_and_normalises_items():
    n = parse_notes_json(RAW)
    assert n.summary == "We planned the launch."
    assert n.decisions == ["Ship on Monday"]
    assert n.action_items[0] == {"owner": "Asha", "task": "Send notes", "due": "Friday"}
    assert n.action_items[1] == {"owner": "Unassigned", "task": "Book the room", "due": None}
    assert len(n.action_items) == 2  # empty task dropped


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_notes_json("sorry, I cannot do that")


def test_chunk_text_respects_limit_and_keeps_lines():
    text = "\n".join(f"[00:{i:02d}] You: " + "word " * 10 for i in range(10))
    chunks = chunk_text(text, 35)
    assert len(chunks) > 1
    assert "\n".join(chunks) == text


def test_markdown_sections():
    notes = Notes("Overview.", ["Point"], ["Decision"], [{"owner": "Asha", "task": "Do it", "due": "Fri"}])
    md = notes_to_markdown("Standup", "28 Oct 2026, 10:00", 125, notes, "[00:01] You: hi")
    assert "# Standup" in md and "02:05 long" in md
    assert "- [ ] **Asha**: Do it (due: Fri)" in md
    assert "## Transcript" in md


def test_markdown_without_summary():
    md = notes_to_markdown("Standup", "now", 10, None, "[00:01] You: hi")
    assert "No summary generated" in md
