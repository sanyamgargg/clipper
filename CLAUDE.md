# Clipper — project context for Claude Code

Read `PLAN.md` before doing anything. It is the full spec and build order.

## What this is
A local pipeline that turns long-form videos from **paid clipping campaigns** (Whop Content Rewards, Vyro, etc.)
into vertical short clips for YouTube Shorts and Instagram Reels. The owner gets paid per 1,000 views by the campaign.
No commentary, no AI voice: clean crop, captions, hook text. Only content from campaigns we have joined (permission granted).

## Owner and constraints
- Owner: Sanyam, full-stack dev (FastAPI + React). Prefers practical, minimal solutions.
- Machine now: **MacBook Air M1, 8 GB RAM**. Later may move to a Windows i5 16 GB laptop (likely Intel iGPU, no NVIDIA).
- Volume: **2 source videos per day**. Pipeline runs overnight.
- Claude usage: **only** the clip-picking step, via `claude -p` on a Claude Pro subscription
  (draws from the monthly Agent SDK / `claude -p` credit, ~$20). Everything else is plain local code.

## Hard rules
1. **One heavy step at a time.** Never run Whisper, rendering and anything else in parallel. 8 GB RAM.
2. **No local LLMs on the M1.** Clip picking goes through `claude -p` only.
3. **Never set or read `ANTHROPIC_API_KEY`.** If set, Claude Code bills the API instead of the subscription.
4. **All machine-specific choices live in `config.yaml`** (whisper backend/model, video encoder, paths).
   Code reads config; never hardcode `h264_videotoolbox`, model names or absolute paths.
5. **Each stage is a pure-ish function with a file/DB contract** (see PLAN.md §4) so stages can be rerun independently.
6. **State in SQLite (`data/clipper.db`)**, media in `data/`. Pipeline must be resumable after a crash.
7. **No Docker** on the Mac (no Metal GPU access). Use a Python venv + `scripts/setup.sh`.
8. Do not automate the Whop / campaign dashboards (no scraping/browser bots). Link submission stays manual.
9. Delete source videos after their clips are rendered (disk space).

## Stack
Python 3.11+, yt-dlp, whisper.cpp (Metal) via `pywhispercpp` or CLI, MediaPipe (face detection), ffmpeg,
SQLite, FastAPI + small React (Vite) review UI, launchd for scheduling, `claude -p` for clip picking.

## Conventions
- Python: type hints, `pathlib`, `logging` (not print), small modules. `ruff` for lint/format.
- Every stage logs start/end, duration and errors to the `runs` table.
- Log Claude calls (tokens, cost if available, model, exit code) to the `llm_calls` table.
- Tests: pytest for pure logic (timestamp mapping, crop smoothing, caption chunking, JSON validation).
