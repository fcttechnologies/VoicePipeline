# FFmpeg

Used for normalization and clip extraction.

## Verify

```bash
/opt/homebrew/bin/ffmpeg -version
/opt/homebrew/bin/ffprobe -version
```

## Normalize

```bash
/opt/homebrew/bin/ffmpeg -y -i source.wav \
  -vn -ac 1 -ar 24000 \
  -af "loudnorm=I=-20:TP=-1.5:LRA=11" \
  normalized.wav
```

## Extract Segment

```bash
/opt/homebrew/bin/ffmpeg -y \
  -ss START_SECONDS -to END_SECONDS \
  -i normalized.wav \
  -ac 1 -ar 24000 \
  -af "loudnorm=I=-20:TP=-1.5:LRA=11" \
  clip.wav
```

## Failure Handling

- If normalization or extraction fails, leave the run folder in place and stop.
- Do not advance `state/processed.json` until clips are written.
