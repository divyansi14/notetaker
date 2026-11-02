# notetaker

Records a meeting on Windows (your mic + the other people's audio from your speakers), transcribes it
locally with faster-whisper, and uses an LLM to write a summary, decisions and action items as Markdown.

## Pipeline

```
mic (sounddevice) ----------> track "You"    \
                                               > faster-whisper (int8, VAD) -> merge by timestamp
system audio (WASAPI loopback) -> track "Others" /                                   |
                                                       transcript -> LLM (chunked, retry) -> notes/*.md
```

Recording the two sides as separate tracks labels "You" vs "Others" without a diarization model.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m notetaker devices          # find your mic / speakers
$env:XAI_API_KEY = "your-key"        # xAI API key from console.x.ai (summaries only)
```

The default summary provider is Grok via xAI. Create an API key at groq, then set `GROQ_API_KEY` in the same PowerShell window where you run `python -m notetaker record`. A Grok app subscription is separate from API access. To use Groq instead, set `llm_provider` to `groq` and `llm_model` to a supported Groq model in `config.json`, then set `$env:GROQ_API_KEY`. Do not paste API keys into chat or commit them to a file.

If xAI returns HTTP 403, check that the API key is enabled, belongs to the selected team in xAI Console, and that the team has API access and available credits. xAI documents 403 as a key/team permission or access restriction.

The default mic is the Windows default input device. Set `mic_device` in `config.json` to a name fragment or device index from `devices` if you want a different mic. Set `loopback_device` to select a different speaker output for system audio. Leave it blank to use the Windows default speaker.

The first recording downloads the selected faster-whisper model (default: `base.en`), so allow time and an internet connection on first use. Transcription runs locally on the CPU. The Groq API key is optional: without it, the transcript is saved without an AI summary.

## Use

```powershell
python -m notetaker record --title "Standup"      # press Enter to stop
python -m notetaker record --no-system --title "Mic test"  # mic only
python -m notetaker process meeting.wav           # existing 16 kHz mono WAV
python -m notetaker process you.wav --system others.wav
```

Notes are written to `notes/`, raw audio to `recordings/` (both git-ignored).

## Tests

```powershell
python -m pytest -q
```

## Known limitations

- Use **headphones**: with speakers, your mic also records the remote voices and they appear under "You".
- "Others" is one label; separating several remote speakers needs diarization (planned: pyannote).
- Non-WAV or non-16 kHz files are decoded by PyAV; if that fails run `pip install -U av` or convert to WAV.
- No summary is generated without the API key for the selected provider (`XAI_API_KEY` by default); the transcript is still saved.
- Long recordings are held in memory (about 115 MB per hour per track).
