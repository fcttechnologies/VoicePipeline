# VoicePipeline

End-to-end pipeline for building a custom F5-TTS voice model from rights-cleared source audio. Pulls source media, normalizes it, transcribes with word-level timestamps, extracts speaker-isolated clips into an F5-TTS-ready dataset, and finetunes the model locally on Apple Silicon or any CPU/CUDA host.

Built and maintained by [FCT Technologies LLC](https://fct-technologies.com). Used internally to ship custom voice models for client products that need a branded, owned voice (IVR, in-app assistants, narration, audio guides). The reference implementation runs on a Mac mini M4 with no GPU dependency.

> **Bring your own audio.** This repository is the pipeline only. No training data, no trained checkpoints, and no copyrighted audio is included or distributed. You are responsible for ensuring you have the rights to every source you feed in.

---

## What it does

1. **Acquire.** Downloads source audio from YouTube via `yt-dlp` (or accepts local files via the drop-zone folder).
2. **Normalize.** Converts to 24 kHz mono WAV with EBU R128 loudness normalization via `ffmpeg`.
3. **Transcribe.** Runs `faster-whisper` directly with `word_timestamps=True` to produce segment- and word-level timing.
4. **Extract dataset clips.** Splits the source into short speaker-isolated clips with text sidecars, writing both `metadata.csv` (F5-TTS format) and per-source provenance manifests.
5. **Pick a reference clip.** Auto-selects the longest extracted clip as the voice profile reference (used for both zero-shot inference and finetuning).
6. **Train.** Prepares the dataset into F5-TTS's `raw.arrow` / `duration.json` / `vocab.txt` format, then invokes `f5-tts_finetune-cli` against the `F5TTS_v1_Base` checkpoint.
7. **Notify.** Sends a Telegram digest at terminal state (done or blocked).

## Why this shape

- **Pipeline-as-asset.** Each stage is idempotent and resumable. Failures in download, normalization, or transcription don't poison the dataset; the source is just not marked processed.
- **No GPU required.** Validated end-to-end on Apple Silicon CPU. MPS backend has a known silent-output bug as of PyTorch 2.12.0 — pipeline defaults to CPU.
- **State-driven, not script-driven.** Source material, processed registry, and run log are all JSON files; the runner is stateless.
- **Manual cleanup gate.** F5-TTS finetune quality is dominated by dataset purity. The pipeline supports a `--train-only` mode so you can run dataset extraction, manually prune false positives, then train against the cleaned set.

## Requirements

- Python 3.11+
- macOS / Linux (validated on macOS 25.5, M4 Apple Silicon)
- `yt-dlp`, `ffmpeg`, `ffprobe` on `PATH`
- ~10 GB free for the F5-TTS environment, base checkpoint, and a small dataset
- Optional: a Telegram bot token + chat ID for notifications

See [setup.md](setup.md) for the full environment setup, including the two virtual environments (`.venv` for the pipeline, `.venv-f5` for F5-TTS).

## Quick start

```bash
# 1. Set up environments (see setup.md for full detail)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python3 -m venv .venv-f5
.venv-f5/bin/pip install f5-tts

# 2. Configure sources (replace placeholder with your rights-cleared media)
cp state/source-material.example.json state/source-material.json
# edit state/source-material.json with your sources

# 3. Build the dataset (acquire + normalize + transcribe + extract, no training)
.venv/bin/python scripts/run_pipeline.py --skip-training

# 4. Manually review datasets/v001/wavs/ — Quick Look through the clips, delete false positives

# 5. Train on the cleaned dataset
.venv/bin/python scripts/run_pipeline.py --train-only
```

## Modes

| Flag | What it does |
|---|---|
| (none) | Full run: pick the next unprocessed source, run all stages, finetune. |
| `--skip-training` | Build dataset only; do not finetune. |
| `--train-only` | Skip source acquisition; finetune against the existing `datasets/v<NNN>` dataset. Use after manual pruning. |
| `--smoke-test` | Short download window, single clip, skips full training. Verifies the environment is wired correctly. |

## Configuration

Override the output and state locations with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `VOICE_PIPELINE_OUTPUT_ROOT` | `./output/` | Where `runs/`, `datasets/`, `voice-models/`, `drop-zone/` are written. |
| `VOICE_PIPELINE_STATE_DIR` | `./state/` | Where `source-material.json`, `processed.json`, `run-log.json` live. |
| `VOICE_PIPELINE_WHISPER_MODEL` | `base.en` | `faster-whisper` model name. |
| `VOICE_PIPELINE_FFMPEG` / `VOICE_PIPELINE_FFPROBE` / `VOICE_PIPELINE_YT_DLP` | from `PATH` | Override binary paths. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | unset | Enables Telegram digests at run completion. |

## Output layout

```text
$VOICE_PIPELINE_OUTPUT_ROOT/
├── runs/<run_id>/                      raw per-run artifacts (source.wav, transcript-segments.json)
├── datasets/v<NNN>/
│   ├── wavs/                           extracted speaker-isolated clips + paired .txt sidecars
│   ├── metadata.csv                    F5-TTS format: <abs_path>|<text>
│   ├── metadata-relative.csv           same, but with wavs/<name> paths
│   ├── manifests/                      per-source provenance JSON
│   └── voice-profile.json              reference clip pointer for inference + finetune
├── voice-models/v<NNN>/                training logs and finetuned checkpoint outputs
└── drop-zone/                          inbox for local file sources (place WAV/MP3 here)
```

## State files

`state/source-material.json` — declared sources (committed config you maintain).
`state/processed.json` — registry of which sources have been ingested into which dataset version (per-machine, gitignored).
`state/run-log.json` — append-only log of every pipeline run (per-machine, gitignored).

## Known limitations

- **CPU-only on Apple Silicon.** The MPS backend produces silent output with PyTorch 2.12.0 + F5-TTS 1.1.20. CPU finetuning of a ~15-minute corpus takes several hours on an M4 Mac mini. Watch upstream PyTorch for the MPS fix.
- **`base.en` Whisper model by default.** Good enough for clean English studio audio. Switch to `large-v3` via `VOICE_PIPELINE_WHISPER_MODEL` for noisier sources at the cost of transcription time.
- **No automatic speaker diarization.** The pipeline assumes the source compilation is mostly the target speaker. Multi-speaker sources require either pre-filtered compilations or a manual review pass before training.

## License

[MIT](LICENSE). The pipeline is open. The audio you feed in and the model you train are your responsibility.

## Built by

[FCT Technologies LLC](https://fct-technologies.com) — software, AI integrations, and automation systems for small businesses and shipped product work.
