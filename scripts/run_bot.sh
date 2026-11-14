#!/usr/bin/env bash
# Telegram bot (long polling). Kept alive by launchd (com.sanyam.clipper.telegram).
# Its log is data/logs/telegram.log (rotated); the bot token comes from the Keychain.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
unset ANTHROPIC_API_KEY
exec "$PWD/.venv/bin/python" -m clipper.telegram bot
