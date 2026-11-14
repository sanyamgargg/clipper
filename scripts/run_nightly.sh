#!/usr/bin/env bash
# Nightly pipeline run. Started by launchd (com.sanyam.clipper, 01:30); also fine by hand.
#   scripts/run_nightly.sh           run the pipeline under caffeinate -i
#   scripts/run_nightly.sh --check   check tools, Claude token and login; no pipeline run
#
# Claude auth: the long-lived token from `claude setup-token`, stored in the Keychain
# (service = claude.keychain_service in config.yaml), is exported as
# CLAUDE_CODE_OAUTH_TOKEN. Without it, the interactive `claude` login is used.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
VPY="$ROOT/.venv/bin/python"

# launchd starts jobs with a minimal PATH; claude lives in ~/.local/bin, the rest in Homebrew
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
unset ANTHROPIC_API_KEY # would make claude bill the API instead of the subscription

cfg() {
  "$VPY" -c 'import sys
from clipper.config import load_config
c = load_config()
print({"logs": c.dir("logs"), "service": c.claude.keychain_service}[sys.argv[1]])' "$1"
}

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

LOG_DIR="$(cfg logs)"
if [[ $CHECK -eq 0 && ! -t 1 ]]; then
  # under launchd: one log file per night
  exec >>"$LOG_DIR/nightly-$(date +%F).log" 2>&1
fi
echo "=== $(date '+%F %T') run_nightly.sh ${1:-}"

SERVICE="$(cfg service)"
if TOKEN="$(security find-generic-password -s "$SERVICE" -w 2>/dev/null)" && [[ -n "$TOKEN" ]]; then
  export CLAUDE_CODE_OAUTH_TOKEN="$TOKEN"
  echo "claude: long-lived token from Keychain ($SERVICE)"
else
  echo "WARNING: no Claude token in Keychain ($SERVICE); using the interactive login," \
    "which expires. Run: claude setup-token" >&2
fi
unset TOKEN

if [[ $CHECK -eq 1 ]]; then
  fail=0
  for tool in claude ffmpeg whisper-cli deno caffeinate; do
    if path="$(command -v "$tool")"; then echo "ok   $tool: $path"; else echo "MISSING $tool"; fail=1; fi
  done
  "$VPY" -m yt_dlp --version >/dev/null && echo "ok   yt-dlp: $("$VPY" -m yt_dlp --version)"
  if [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
    # empty HOME: no stored login to fall back on, so this only passes with the token
    tmp_home="$(mktemp -d)"
    if reply="$(HOME="$tmp_home" claude -p "Reply with exactly: ok" --model haiku \
      --max-turns 1 --tools "" </dev/null 2>&1)"; then
      echo "ok   claude -p with the Keychain token replied: $reply"
    else
      echo "FAIL claude -p with the Keychain token: $(tail -c 300 <<<"$reply")"
      fail=1
    fi
    rm -rf "$tmp_home"
  else
    fail=1
  fi
  exit $fail
fi

exec caffeinate -i "$VPY" -m clipper.run
