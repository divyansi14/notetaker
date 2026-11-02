import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Config:
    model: str = "base.en"
    language: str = "en"
    beam_size: int = 1
    cpu_threads: int = 0
    compute_type: str = "int8"
    vocabulary: list = field(default_factory=list)  # names/jargon Whisper should expect
    mic_device: str = ""            # name fragment or index; empty = Windows default mic
    capture_system: bool = True     # also record system audio (the other people on a call)
    loopback_device: str = ""       # speaker name fragment; empty = default speakers
    llm_base_url: str = "https://api.groq.com/openai/v1"  # any OpenAI-compatible API
    llm_api_key_env: str = "GROQ_API_KEY"                 # env var holding that API's key
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: float = 60.0
    chunk_words: int = 1500         # long transcripts are summarised in chunks of this size
    recordings_dir: str = "recordings"
    notes_dir: str = "notes"


def load_config(path: Path = CONFIG_PATH) -> Config:
    cfg = Config()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg