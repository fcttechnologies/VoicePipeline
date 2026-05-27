# Transcription

This automation uses Faster-Whisper directly because it needs segment and word timestamps.

## Local TranscribingApp Check

TranscribingApp runs at:

```text
http://127.0.0.1:8000/
```

Its current API returns plain text for `/api/transcribe/file` and job results. That is useful for human transcripts but not enough for clip extraction. Do not modify TranscribingApp for v1.

## Runtime

Use the automation venv:

```bash
.venv/bin/python -c "import faster_whisper; print('ok')"
```

Default model:

```text
base.en
```

Override:

```bash
VOICE_PIPELINE_WHISPER_MODEL=small.en .venv/bin/python scripts/run_pipeline.py --skip-training
```

## Timestamp Call

The runner uses:

```python
segments, info = model.transcribe(
    "normalized.wav",
    beam_size=5,
    vad_filter=False,
    word_timestamps=True,
)
```

## Failure Handling

- Empty timestamped output is a blocker.
- Do not mark the source processed on transcription failure.
