import argparse
import os
from datetime import datetime

from .config import ROOT, load_config
from .pipeline import load_audio, run_pipeline, slug


def load_env():
    """Load KEY=VALUE lines from .env into the environment (a real env var wins)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def show_notes(path):
    """Print everything above the transcript so the summary is visible in the terminal."""
    text = path.read_text(encoding="utf-8")
    print("\n" + text.split("## Transcript")[0].strip() + "\n")
    print(f"Notes saved to {path}")


def cmd_devices(cfg):
    import sounddevice as sd

    print("Input devices (set mic_device to a name fragment or index):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']}")
    try:
        import soundcard as sc

        print("\nSpeakers (loopback sources; set loopback_device to a name fragment):")
        for s in sc.all_speakers():
            print(f"  {s.name}")
        print(f"Default speaker: {sc.default_speaker().name}")
    except Exception as exc:
        print(f"\nsoundcard unavailable: {exc}")


def cmd_record(args, cfg):
    from .capture import LoopbackTrack, MicTrack, save_wav

    mic = MicTrack(cfg.mic_device)
    system = None
    if cfg.capture_system and not args.no_system:
        try:
            system = LoopbackTrack(cfg.loopback_device)
            system.start()
            print(f"System audio: recording from '{system.speaker_name}'")
        except Exception as exc:
            print(f"System audio unavailable ({exc}); recording mic only.")
            system = None

    started = datetime.now()
    mic.start()
    input("Recording ... press Enter to stop. ")
    tracks = {"You": mic.stop()}
    if system:
        tracks["Others"] = system.stop()

    rec_dir = ROOT / cfg.recordings_dir
    rec_dir.mkdir(exist_ok=True)
    stamp = f"{started:%Y-%m-%d_%H%M}-{slug(args.title)}"
    for speaker, audio in tracks.items():
        save_wav(rec_dir / f"{stamp}_{speaker.lower()}.wav", audio)
    print(f"Saved recordings to {rec_dir}")

    show_notes(run_pipeline(tracks, args.title, cfg, started))


def cmd_process(args, cfg):
    if args.system:
        tracks = {"You": load_audio(args.file), "Others": load_audio(args.system)}
    else:
        tracks = {"Speaker": load_audio(args.file)}
    show_notes(run_pipeline(tracks, args.title, cfg))


def main():
    parser = argparse.ArgumentParser(prog="notetaker", description="Record meetings, get notes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="list microphones and speakers")

    rec = sub.add_parser("record", help="record a meeting, then transcribe and summarise")
    rec.add_argument("--title", default="Meeting")
    rec.add_argument("--no-system", action="store_true", help="mic only, skip system audio")

    proc = sub.add_parser("process", help="transcribe and summarise existing audio")
    proc.add_argument("file", help="audio file (your mic track, or the only track)")
    proc.add_argument("--system", help="second audio file for the remote participants")
    proc.add_argument("--title", default="Meeting")

    args = parser.parse_args()
    load_env()
    cfg = load_config()
    if args.cmd == "devices":
        cmd_devices(cfg)
    elif args.cmd == "record":
        cmd_record(args, cfg)
    else:
        cmd_process(args, cfg)