# Telegram

Optional terminal-state notifier. The pipeline sends a digest at run completion if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set; otherwise it silently skips.

## Environment

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot API token from `@BotFather` |
| `TELEGRAM_CHAT_ID` | Destination chat ID (your DM with the bot, or a group) |

## Send Message

```bash
curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d chat_id="$TELEGRAM_CHAT_ID" \
  -d text="message here"
```

## Digest Shape

```text
Voice Pipeline - [done|blocked]
Run: [run_id]
Source: [title]
Clips: [count]
Training: [complete|skipped|failed]
Output: [run path]
[message]
```

If credentials are missing, the digest is printed to stdout and the run still succeeds.
