#!/usr/bin/env bash
# M0 setup for macOS (Apple Silicon). Idempotent: safe to rerun.
#   scripts/setup.sh              full setup + checks
#   scripts/setup.sh --no-claude  skip the `claude -p` check (it uses a little subscription credit)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CHECK_CLAUDE=1
[[ "${1:-}" == "--no-claude" ]] && CHECK_CLAUDE=0

say_step() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. brew deps -----------------------------------------------------------
# ffmpeg-full, not ffmpeg: the slim `ffmpeg` formula has no libass (needed to burn captions).
# deno: JavaScript runtime yt-dlp needs to solve YouTube's download challenges.
say_step "Homebrew packages (ffmpeg-full, whisper-cpp, deno)"
command -v brew >/dev/null || die "Homebrew not found. Install from https://brew.sh"
for pkg in ffmpeg-full whisper-cpp deno; do
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    echo "$pkg already installed"
  else
    brew install "$pkg"
  fi
done
# Capture output first: `ffmpeg | grep -q` trips pipefail when grep exits early (SIGPIPE).
has_ass() {
  command -v ffmpeg >/dev/null || return 1
  local filters; filters="$(ffmpeg -hide_banner -filters 2>/dev/null)"
  grep -q ' ass ' <<<"$filters"
}
if ! has_ass; then
  # ffmpeg-full is keg-only; put it on PATH in place of the slim build (reversible with brew link).
  brew unlink ffmpeg >/dev/null 2>&1 || true
  brew link --force ffmpeg-full
  hash -r
fi
has_ass || die "ffmpeg on PATH still has no 'ass' filter: $(command -v ffmpeg)"

# --- 2. python venv ---------------------------------------------------------
say_step "Python venv (.venv)"
PY=""
for cand in python3.12 python3.11; do
  if command -v "$cand" >/dev/null; then PY="$cand"; break; fi
done
[[ -n "$PY" ]] || die "Need Python 3.11 or 3.12 (mediapipe has no wheels for newer). brew install python@3.11"
[[ -d .venv ]] || "$PY" -m venv .venv
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -e ".[dev]"
echo "venv: $("$VPY" --version)"

# --- 3. config --------------------------------------------------------------
say_step "config.yaml"
if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "created config.yaml from config.example.yaml (edit campaigns before the first run)"
else
  echo "config.yaml exists"
fi
cfg() { "$VPY" -c 'import sys,yaml; d=yaml.safe_load(open("config.yaml"))
for k in sys.argv[1].split("."): d=d[k]
print(d)' "$1"; }
DATA_DIR="$(cfg machine.data_dir)"
WHISPER_MODEL="$(cfg machine.whisper_model)"
CLAUDE_MODEL="$(cfg claude.model)"
mkdir -p "$DATA_DIR"/{sources,transcripts,clips,logs,models}

# --- 4. whisper model -------------------------------------------------------
say_step "Whisper model ($WHISPER_MODEL)"
MODEL_FILE="$DATA_DIR/models/ggml-$WHISPER_MODEL.bin"
if [[ -f "$MODEL_FILE" ]]; then
  echo "model present: $MODEL_FILE"
else
  curl -fL --retry 3 -C - -o "$MODEL_FILE.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WHISPER_MODEL.bin"
  mv "$MODEL_FILE.part" "$MODEL_FILE"
fi

# --- 5. checks --------------------------------------------------------------
say_step "Checks"
ffmpeg -hide_banner -version | head -1
"$VPY" -m yt_dlp --version | sed 's/^/yt-dlp /'
"$VPY" -c 'import mediapipe, fastapi, pydantic; print("python deps OK (mediapipe", mediapipe.__version__ + ")")'

ENCODER="$(cfg machine.video_encoder)"
ENCODERS="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
grep -q " $ENCODER " <<<"$ENCODERS" \
  || die "ffmpeg has no encoder '$ENCODER' (machine.video_encoder in config.yaml)"
echo "encoder $ENCODER available"

# Whisper smoke test: synthesize a sentence, transcribe it.
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
say -o "$TMP/t.aiff" "The quick brown fox jumps over the lazy dog."
ffmpeg -hide_banner -loglevel error -i "$TMP/t.aiff" -ar 16000 -ac 1 "$TMP/t.wav"
OUT="$(whisper-cli -m "$MODEL_FILE" -f "$TMP/t.wav" -nt 2>/dev/null | tr -d '\n')"
echo "whisper heard:$OUT"
grep -qi "fox" <<<"$OUT" || die "whisper smoke test failed"

if [[ $CHECK_CLAUDE -eq 1 ]]; then
  say_step "claude -p (subscription login)"
  command -v claude >/dev/null || die "claude CLI not found"
  [[ -z "${ANTHROPIC_API_KEY:-}" ]] \
    || die "ANTHROPIC_API_KEY is set: claude would bill the API, not the subscription. Unset it."
  REPLY="$(claude -p "Reply with exactly: ok" --model "$CLAUDE_MODEL" --max-turns 1 </dev/null)" \
    || die "claude -p failed. Run 'claude' and /login with the Pro subscription."
  echo "claude replied: $REPLY"
fi

say_step "M0 setup complete"
