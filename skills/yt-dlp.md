# yt-dlp

Used to acquire public YouTube dialogue-compilation audio.

## Search

```bash
yt-dlp "ytsearch5:voice dialogue compilation" \
  --skip-download \
  --print "%(title)s\t%(webpage_url)s\t%(duration_string)s"
```

## Download Audio

Full source:

```bash
yt-dlp --no-playlist --force-overwrites \
  --extract-audio --audio-format wav --audio-quality 0 \
  --ffmpeg-location /opt/homebrew/bin \
  -o "source.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

Smoke sample:

```bash
yt-dlp --no-playlist --force-overwrites \
  --extract-audio --audio-format wav --audio-quality 0 \
  --ffmpeg-location /opt/homebrew/bin \
  --download-sections "*00:00-20" \
  -o "source.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Failure Handling

- YouTube JS runtime warnings are non-fatal if audio still downloads.
- On extraction failure, do not mark the source as processed.
- Keep the run folder for inspection.
