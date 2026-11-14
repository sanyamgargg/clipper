# Clipper — Full Plan & Spec

## 1. Goal

Earn per-view payouts from **clipping campaigns** by posting short vertical clips of campaign content
to YouTube Shorts and Instagram Reels, with near-zero daily effort.

- Income model: campaigns pay roughly **$0.50–$3 per 1,000 approved views** (median ~$1). Payment only after the
  campaign owner approves the clip; campaigns have budgets and per-clip caps; marketplace takes a fee (~10%).
- We only clip content from campaigns we have joined. No unlicensed reposting.
- **This is an experiment.** If the go/no-go test (§11) fails, the project is dropped.

### Daily operating target
| When | What | Who | Time |
|---|---|---|---|
| Night | Fetch → transcribe → pick → render (2 source videos) | Pipeline | 0 min |
| Morning | Approve/reject candidate clips in review UI | Sanyam | ~10 min |
| Day | Post approved clips (manual in v1, API in v2) | Sanyam / pipeline | ~2 min |
| Evening | Submit post links to campaigns | Sanyam | ~3 min |
| Weekly | Swap campaigns out of budget, update config | Sanyam | ~10 min |

## 2. Out of scope (do NOT build)
- Commentary, voice cloning, TTS, AI avatars.
- Multiple accounts per platform / view manipulation.
- Scraping or browser-automating Whop / campaign dashboards.
- Local LLMs on the M1.
- Docker.

## 3. Hardware & environment

| | Now: MacBook Air M1 8 GB | Later: Windows i5 16 GB (likely Intel iGPU) |
|---|---|---|
| Whisper | whisper.cpp + Metal, `large-v3-turbo` q5 | faster-whisper, CPU int8, `medium` or `large-v3-turbo` |
| Encoder | `h264_videotoolbox` | `h264_qsv` (fallback `libx264 -preset veryfast`) |
| Clip picking | `claude -p` | `claude -p` (optionally compare with local 7–8B later) |
| Scheduler | launchd + `caffeinate -i` | Task Scheduler |

Memory rules (8 GB): strictly sequential stages, free models after use, process one video at a time,
keep working files in `data/`, delete source video after render.

## 4. Architecture

```
clipper/
├── CLAUDE.md
├── PLAN.md
├── config.example.yaml        # copy to config.yaml (gitignored)
├── pyproject.toml
├── scripts/
│   ├── setup.sh               # brew deps, venv, whisper model download
│   └── run_nightly.sh         # caffeinate -i python -m clipper.run
├── launchd/com.sanyam.clipper.plist
├── clipper/
│   ├── config.py              # load + validate config.yaml (pydantic)
│   ├── db.py                  # SQLite schema + helpers
│   ├── run.py                 # orchestrator: runs stages for pending work
│   ├── stages/
│   │   ├── fetch.py
│   │   ├── transcribe.py
│   │   ├── pick.py
│   │   ├── render.py
│   │   └── publish.py         # phase 2
│   ├── render/
│   │   ├── facetrack.py       # MediaPipe detection + smoothing → crop path
│   │   ├── captions.py        # word-highlight ASS subtitle generation
│   │   └── ffmpeg.py          # command builder (encoder from config)
│   └── llm/claude_cli.py      # wraps `claude -p`, JSON validation, retries, logging
├── prompts/pick_clips.md
├── review/
│   ├── api.py                 # FastAPI: list clips, approve/reject, mark posted, add post URL
│   └── web/                   # React (Vite) single page
├── tests/
└── data/                      # gitignored: sources/, transcripts/, clips/, clipper.db, logs/
```

### Stage contracts

| Stage | Input | Output | Status after |
|---|---|---|---|
| fetch | `campaigns[].sources` in config | `data/sources/<video_id>.mp4`, row in `videos` | `downloaded` |
| transcribe | source mp4 | `data/transcripts/<video_id>.words.json` (word-level) + `.compact.txt` | `transcribed` |
| pick | compact transcript + campaign rules | rows in `clips` (start, end, hook, title, score) | `picked` |
| render | source mp4 + clip rows + word JSON | `data/clips/<clip_id>.mp4` | clips `rendered`, video `done`, source deleted |
| publish (v2) | approved clips | platform post IDs/URLs | clips `posted` |

Orchestrator: for each stage, find items in the previous status and process them one at a time.
Any failure → mark `failed` with error text, continue with the next item. Rerunnable at any time.

## 5. Config (`config.yaml`)

```yaml
machine:
  whisper_backend: whisper_cpp        # whisper_cpp | faster_whisper
  whisper_model: large-v3-turbo-q5_0
  video_encoder: h264_videotoolbox    # h264_videotoolbox | h264_qsv | libx264
  data_dir: ./data

pipeline:
  videos_per_night: 2
  candidates_per_video: 5
  clip_min_sec: 20
  clip_max_sec: 45

claude:
  model: sonnet          # sonnet | haiku  (switch to haiku if credit runs low)
  max_turns: 3
  timeout_sec: 300

campaigns:
  - id: example-podcast
    name: "Example Podcast"
    rate_per_1k: 1.0
    platforms: [youtube_shorts, instagram_reels]
    sources:
      - https://www.youtube.com/@example/videos   # channel or playlist given by the campaign
    rules:
      min_sec: 15
      max_sec: 60
      required_caption_text: ""                   # e.g. "@creator" if campaign requires it
      notes: "No watermarks. Must credit creator in caption."
    active: true
```

## 6. Stage details

### 6.1 fetch
- yt-dlp, max 1080p mp4, skip already-known IDs (`videos` table).
- Only pull the newest N per night across active campaigns (`videos_per_night`).
- Save metadata (title, duration, upload date, campaign_id).

### 6.2 transcribe
- whisper.cpp with Metal; word-level timestamps.
- Write `words.json`: `[{"w": "hello", "s": 12.34, "e": 12.61}, ...]`.
- Write `compact.txt`: one line per sentence/segment: `[mm:ss] text` — this is what Claude sees (3–5× fewer tokens).
- Unload the model when done.

### 6.3 pick (the only Claude step)
- Build prompt from `prompts/pick_clips.md` + campaign rules + compact transcript.
- Call: `claude -p --model <cfg> --output-format json --max-turns <cfg>` (prompt via stdin).
- Parse and validate against the JSON schema in the prompt; on invalid JSON retry once with a "fix your JSON" message.
- Map each clip's `start`/`end` to the nearest word boundaries in `words.json`; enforce min/max length;
  never cut mid-word or mid-sentence.
- Non-zero exit (e.g. credit exhausted) → leave video in `transcribed`, log, stop the stage for tonight.
- Log to `llm_calls`.

### 6.4 render
For each picked clip:
1. **Face tracking:** sample frames at ~5 fps with MediaPipe, find primary face center, smooth with
   EMA/one-euro filter, clamp to frame, interpolate to a crop-x path. No face → center crop.
   For two-person podcasts: follow the active speaker if easy, else the largest face; avoid jittery switching
   (minimum 1.5 s between switches).
2. **Crop & scale** to 1080×1920, 30 fps.
3. **Captions:** ASS subtitles, 2–4 words per chunk, current word highlighted, bold sans font, outline,
   positioned around 60–70% of height (out of the bottom UI safe zone and away from the top).
4. **Hook text:** first ~2.5 s, top area, from Claude's `hook_text`.
5. **Audio:** loudness normalize to about -14 LUFS.
6. Encoder from config; target ~8–10 Mbps; output `data/clips/<clip_id>.mp4`.
7. Also write a thumbnail JPG for the review UI.
8. After all clips of a video render OK → delete the source mp4.

### 6.5 review UI
- FastAPI + React page listing rendered clips (newest first): video player, hook, title, score, campaign.
- Buttons: Approve / Reject. Approved: copy-to-clipboard title + caption (with required campaign text),
  download button, field to paste the posted URL per platform.
- Runs locally: `uvicorn review.api:app`, open on phone via LAN for morning review.

### 6.6 publish (phase 2 — only after views look promising)
- **YouTube Data API v3** `videos.insert`: needs a Google Cloud project; unverified projects can only upload
  **private** videos until the app passes Google's audit; upload costs a large chunk of daily quota (~6 uploads/day).
- **Instagram Graph API**: needs an Instagram Business/Creator account linked to a Facebook Page and a Meta app.
  Reels publishing takes a **publicly reachable `video_url`**, so clips need temporary public hosting
  (e.g. an S3 bucket with a pre-signed URL) during publish.
- Start the approvals early (1–3 weeks); keep manual posting until they land.
- Schedule posts spaced across the day; add title, hashtags and required campaign caption text.

## 7. Database (SQLite)

```sql
campaigns(id TEXT PK, name, rate_per_1k REAL, active INT)
videos(id TEXT PK, campaign_id, url, title, duration_sec, status, error, created_at, updated_at)
clips(id TEXT PK, video_id, start_sec REAL, end_sec REAL, hook_text, title, reason, score REAL,
      status,  -- picked | rendered | approved | rejected | posted | failed
      file_path, created_at)
posts(id INTEGER PK, clip_id, platform, url, posted_at, views INT, views_checked_at,
      submitted_to_campaign INT, campaign_status, earnings REAL)
runs(id INTEGER PK, stage, item_id, started_at, ended_at, ok INT, error)
llm_calls(id INTEGER PK, video_id, model, input_tokens, output_tokens, cost_usd, exit_code, created_at)
```

## 8. Scheduling
- `scripts/run_nightly.sh`: `cd` to repo, activate venv, `caffeinate -i python -m clipper.run`.
- launchd plist runs it at 01:30 every night; logs to `data/logs/`.
- Mac must be on charger with lid open or "prevent sleep" enabled.

## 9. Claude credit budget (Pro, `claude -p` credit ≈ $20/month)
- 2 videos/day × ~30k tokens per call ≈ **$5/month on Sonnet**, <$1 on Haiku (estimates).
- Keep `--max-turns` low, one call per video, compact transcripts only.
- Check usage after week 1; switch `claude.model` to `haiku` if needed.

## 10. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Setup | `scripts/setup.sh` installs ffmpeg, yt-dlp, whisper.cpp model, venv; `claude -p "hi"` works with subscription login |
| M1 | One video end-to-end (CLI) | `python -m clipper.run --url <video>` produces 3–5 good clips in `data/clips/` |
| M2 | Batch + DB + resumable | Pipeline processes 2 videos from config, survives a kill mid-run |
| M3 | Review UI | Approve/reject + mark posted from phone browser |
| M4 | Nightly schedule | launchd runs it unattended for 3 nights in a row |
| M5 | Metrics | Posted URLs + views + campaign status tracked; weekly summary command |
| M6 | API publishing | YouTube + Instagram auto-post after approvals land |

Build M0–M4 first (≈ 2 weekends). M5–M6 only if the test in §11 looks positive.

## 11. Go / no-go test (first 14 days of posting)

| Result | Decision |
|---|---|
| Any clip > 20k views and payouts approved | ✅ Continue: build M5–M6, consider more campaigns / moving to the 16 GB laptop |
| Steady 1–5k views per clip, some approvals | 🟡 Two more weeks, improve hooks + clip picking prompt |
| Most clips < 500 views or rejected | ❌ Change campaigns/niche once; if still flat, drop the project |

## 12. Risks
- Campaign rejections (rule violations) → per-campaign rules in config, checked in review.
- Platform originality / duplicate detection → only campaign content, native uploads, no watermarks, one account per platform.
- Credit runs out mid-month → pipeline pauses gracefully at `pick`; switch to Haiku.
- 8 GB RAM pressure → sequential stages, overnight runs.
- Payout delays/pending balances on marketplaces → withdraw promptly, track in `posts`.

## 13. First campaign: TJR (Reach, via Content Rewards) — implications
Full rules: https://reachclipping.com/TJR (also stored in `config.yaml` → `campaigns[tjr].rules`).

- **Config schema:** campaign `rules` now has extra fields (`required_hashtags`, `mvr_views`, `submit_within_minutes`,
  `must_feature`, `no_logos`, `look_for`, ...). The config model must accept them, and `pick` must pass
  `look_for`, `must_feature` and `no_negative_portrayal` into the clip-picking prompt.
- **Captions/titles:** always append required hashtags (`#TJR`). English only.
- **No logos/watermarks:** never add our own branding; avoid crops that keep third-party logo overlays when possible.
- **Thumbnails:** a frame from the clip itself, different per clip.
- **30-minute submission rule:** the link must be submitted on Content Rewards within 30 min of posting.
  Publishing (manual or API) must happen only when Sanyam can submit right after. In M6, auto-publish must send a
  notification with the post URL so it can be submitted immediately; never post at night unattended.
- **Payout threshold:** a clip earns only after 5,000 views (MVR). Track views per post against `mvr_views`.
- **Audience risk:** campaign requires 50%+ audience from USA/CA/AU/UK/NZ. Accounts run from India may skew to
  Indian viewers. Check audience geography in YouTube/Instagram analytics after the first week.
- **Sources:** YouTube @TJRTrades (videos + streams) and @TRichesTrades. Streams are long: transcribe only, then
  download/cut just the needed segments if disk is tight.
