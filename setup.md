# Setup

Reference implementation for getting VoicePipeline running on a fresh macOS or Linux host.

## Tool choice

Default TTS engine: **F5-TTS**.

- Local-first. No cloud calls. Works offline once the base checkpoint is downloaded.
- Active package with current CLI inference and CLI finetuning support.
- Short reference clips (5-15s) are enough for zero-shot inference.
- Wraps cleanly behind a LiveKit Agents custom TTS node when you need real-time voice.

Fallbacks (not configured in this repo):

| Tool | When you'd reach for it |
|---|---|
| Coqui XTTS-v2 | F5-TTS quality is poor on your speaker or training is unstable. Older stack but solid zero-shot. |
| ElevenLabs | Cloud-based prototyping when training time is the bottleneck. Paid per call. |
| RVC | Voice conversion, not text-to-speech. Different problem. |

## Prerequisites

| Need | How |
|---|---|
| `yt-dlp` | `uv tool install yt-dlp` or `pip install yt-dlp` |
| `ffmpeg`, `ffprobe` | `brew install ffmpeg` (macOS) or distro package |
| Python 3.11+ for the pipeline runner | system Python or `uv` |
| Python 3.11 for F5-TTS | F5-TTS pins 3.11; do not use 3.12+ for the F5 venv |
| Telegram bot (optional) | Create a bot via `@BotFather`, expose token + chat id as env vars |

## Two virtual environments

VoicePipeline uses two isolated environments because the pipeline runner (faster-whisper, requests, ffmpeg shells) and F5-TTS have different dependency trees.

```bash
# 1. Pipeline runner
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. F5-TTS (must be Python 3.11)
python3.11 -m venv .venv-f5
.venv-f5/bin/pip install f5-tts
```

The runner shells out to `.venv-f5/bin/f5-tts_finetune-cli` and `.venv-f5/bin/python` at training time. If those paths exist, training works. Override them by editing `scripts/run_pipeline.py:f5_paths()`.

## State files

```bash
cp state/source-material.example.json state/source-material.json
# Edit state/source-material.json with your rights-cleared sources
```

`state/processed.json` and `state/run-log.json` are created on first run. They are gitignored — they record per-machine progress and absolute paths.

## Configuration

Set environment variables to override defaults:

```bash
export VOICE_PIPELINE_OUTPUT_ROOT="$HOME/voice-output"       # default: ./output/
export VOICE_PIPELINE_STATE_DIR="/etc/voicepipeline/state"   # default: ./state/
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

Source these in your shell profile, a `.envrc`, or a launchd plist.

## Smoke test

```bash
.venv/bin/python scripts/run_pipeline.py --smoke-test
```

Expected:

- Downloads ~20s of the first unprocessed source via `yt-dlp`.
- Normalizes to 24 kHz mono WAV.
- Transcribes with `faster-whisper`, word timestamps on.
- Extracts one clip.
- Writes a `datasets/smoke/` dataset and a `voice-models/smoke/training-skipped.json` marker.
- Appends a `smoke-*` entry to `state/run-log.json`.

Smoke runs do not update the processed registry, so you can re-run safely.

## Full run

```bash
# Build + finetune in one pass
.venv/bin/python scripts/run_pipeline.py

# Build only, defer training
.venv/bin/python scripts/run_pipeline.py --skip-training

# Train on existing dataset (use after manual review/pruning of datasets/v<NNN>/wavs/)
.venv/bin/python scripts/run_pipeline.py --train-only
```

Full finetuning on CPU is slow. Plan for an overnight run when you commit. On an M4 Mac mini, ~10-15 minutes of training corpus takes roughly 6-12 hours.

## Hardware notes

- **Apple Silicon CPU works.** Validated on Mac mini M4, 16 GB RAM.
- **MPS backend is broken** in current PyTorch + F5-TTS (silent output). Force `--device cpu` for inference. The training CLI runs on CPU by default.
- **CUDA** is faster if you have it. F5-TTS detects and uses GPU automatically.

## Known limits

- Source compilations may include non-target speakers, music, or SFX. Pipeline does not perform speaker diarization. Always manually review `datasets/v<NNN>/wavs/` before a serious finetune.
- Default Whisper model is `base.en` — good for clean English studio audio. Use `VOICE_PIPELINE_WHISPER_MODEL=large-v3` for noisier sources.
- The runner picks the *longest* extracted clip as the voice profile reference. You may want to override this by editing `voice-profile.json` to point at the cleanest clip instead.
