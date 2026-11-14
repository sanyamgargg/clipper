"""Telegram inbox: formatting, delivery, buttons, link replies, linking, API client.
The Bot API is faked; nothing here touches the network."""

import json
from datetime import UTC, datetime, timedelta

import pytest
import requests

from clipper import db
from clipper.config import Campaign, Config, Machine, Rules, Telegram
from clipper.stages.pick import Clip, load_clips, save_clips
from clipper.telegram import api as api_mod
from clipper.telegram import inbox as ib
from clipper.telegram.api import BotAPI, TelegramError, get_token

CHAT = 4242
SUBMIT = "https://contentrewards.com/c/campaigns/abc"


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict | None]] = []
        self.fail: dict[str, Exception] = {}
        self.updates: list[dict] = []
        self.next_id = 100

    def call(self, method, params=None, files=None, timeout=60, retries=3):
        self.calls.append((method, dict(params or {}), files))
        if method in self.fail:
            raise self.fail[method]
        if method in ("sendVideo", "sendMessage"):
            self.next_id += 1
            return {"message_id": self.next_id}
        if method == "getUpdates":
            return self.updates
        return True

    def methods(self) -> list[str]:
        return [c[0] for c in self.calls]

    def last(self, method: str) -> dict:
        return next(p for m, p, _ in reversed(self.calls) if m == method)


def make_clip(n: int, title: str = "Why <rich> people & you") -> Clip:
    return Clip(f"v1_{n:02d}", "v1", 10.0 * n, 10.0 * n + 30, "Why YOU", title,
                f"{title} #TJR", "reason", 0.8)  # fmt: skip


@pytest.fixture
def ix(tmp_path, monkeypatch):
    cfg = Config(
        machine=Machine(data_dir=tmp_path, min_free_gb=0),
        telegram=Telegram(enabled=True, chat_id=CHAT, max_upload_mb=45),
        campaigns=[
            Campaign(
                id="tjr",
                name="TJR <Reach>",
                submit_url=SUBMIT,
                rules=Rules(submit_within_minutes=30),
            )
        ],
    )
    conn = db.connect(tmp_path)
    db.add_video(conn, "v1", "tjr", "https://x/v1", "Video", 600)
    clips = [make_clip(1), make_clip(2, "Second")]
    save_clips(clips, cfg.dir("transcripts") / "v1.clips.json")
    for c in clips:
        db.add_clip(conn, c.id, c.video_id, c.start_sec, c.end_sec, c.hook_text, c.title,
                    c.reason, c.score)  # fmt: skip
        db.mark_clip_rendered(conn, c.id, "x", "crop")
        for suffix in (".mp4", ".jpg", ".json"):
            (cfg.dir("clips") / f"{c.id}{suffix}").write_bytes(b"x" * 10)
    monkeypatch.setattr(ib, "probe_video", lambda path: (1080, 1920, 30.2))
    monkeypatch.setattr(ib, "make_thumbnail", lambda jpg, work: None)
    yield ib.Inbox(cfg, conn, FakeAPI(), CHAT)
    conn.close()


def status(ix, clip_id: str) -> str:
    return db.get_clip(ix.conn, clip_id)["status"]


def press(ix, data: str, chat: int = CHAT) -> None:
    ib.handle_update(
        ix,
        {"update_id": 1, "callback_query": {"id": "cb", "data": data,
                                            "message": {"message_id": 1, "chat": {"id": chat}}}},
    )  # fmt: skip


def send_text(ix, text: str, reply_to: int | None = None, chat: int = CHAT) -> None:
    message = {"message_id": 999, "chat": {"id": chat, "type": "private"}, "text": text}
    if reply_to:
        message["reply_to_message"] = {"message_id": reply_to}
    ib.handle_update(ix, {"update_id": 2, "message": message})


# --- formatting -------------------------------------------------------------------


def test_caption_escapes_html_and_puts_the_post_caption_in_code(ix):
    clip, campaign = ix.clip("v1_01")
    text = ib.clip_caption(clip, campaign)
    assert text.startswith("<b>Why &lt;rich&gt; people &amp; you</b>\nTJR &lt;Reach&gt;")
    assert "score 0.80 · 30 s" in text
    assert "<code>Why &lt;rich&gt; people &amp; you #TJR</code>" in text
    assert f"Submit within 30 min of posting:\n{SUBMIT}" in text
    assert "<rich>" not in text


def test_caption_stays_within_telegram_limit(ix):
    clip, campaign = ix.clip("v1_01")
    long = Clip(**{**clip.__dict__, "title": "word " * 400})
    text = ib.clip_caption(long, campaign, note="✅ Posted 10:00")
    assert len(text) <= 1024
    assert text.endswith("✅ Posted 10:00") and "#TJR</code>" in text


@pytest.mark.parametrize(
    ("url", "platform"),
    [
        ("https://youtube.com/shorts/abc123", "youtube_shorts"),
        ("https://www.youtube.com/shorts/abc?feature=share", "youtube_shorts"),
        ("https://youtu.be/abc", "youtube_shorts"),
        ("https://www.instagram.com/reel/Cx1/", "instagram_reels"),
        ("https://instagram.com/p/Cx1", "instagram_reels"),
        ("https://www.tiktok.com/@me/video/1", "tiktok"),
        ("https://notyoutube.com/x", "other"),
    ],
)
def test_platform_detection(url, platform):
    assert ib.platform_of(url) == platform


def test_find_urls_strips_trailing_punctuation():
    text = "yt: https://youtube.com/shorts/a, ig (https://instagram.com/reel/b)."
    assert ib.find_urls(text) == ["https://youtube.com/shorts/a", "https://instagram.com/reel/b"]


def test_buttons_offer_the_other_layout():
    crop = ib.buttons("v1_01", "crop")["inline_keyboard"]
    assert [b["callback_data"] for b in crop[0]] == ["done:v1_01", "reject:v1_01"]
    assert crop[1][0] == {"text": "🔁 Fit layout", "callback_data": "fit:v1_01"}
    assert ib.buttons("v1_01", "fit")["inline_keyboard"][1][0]["callback_data"] == "crop:v1_01"


# --- size check --------------------------------------------------------------------


def test_upload_bitrate_only_when_over_the_limit():
    assert ib.upload_bitrate_kbps(44_000_000, 45, 45e6) is None
    kbps = ib.upload_bitrate_kbps(51_000_000, 45, 45e6)
    assert kbps == int(45e6 * 8 * 0.95 / 45 / 1000 - 160) == 7440
    assert (kbps + 160) * 1000 * 45 / 8 < 45e6  # the result fits
    assert ib.upload_bitrate_kbps(500_000_000, 45, 45e6) == 7440
    assert ib.upload_bitrate_kbps(10**12, 45, 45e6) == 7440
    assert ib.upload_bitrate_kbps(10**12, 3600, 45e6) == 500  # floor


# --- delivery ----------------------------------------------------------------------


def test_deliver_sends_video_with_caption_buttons_and_metadata(ix):
    assert ib.deliver(ix, limit=10) == (2, 0)
    sends = [(p, f) for m, p, f in ix.api.calls if m == "sendVideo"]
    params, files = sends[0]
    assert params["chat_id"] == CHAT and params["parse_mode"] == "HTML"
    assert params["supports_streaming"] is True
    assert (params["width"], params["height"], params["duration"]) == (1080, 1920, 30)
    assert params["reply_markup"] == ib.buttons("v1_01", "crop")
    assert files["video"][0] == "v1_01.mp4"
    row = db.get_clip(ix.conn, "v1_01")
    assert (row["status"], row["telegram_message_id"]) == ("delivered", 101)
    assert not (ix.cfg.dir("clips") / "v1_01.mp4").exists()  # delete_local_after_send
    assert (ix.cfg.dir("clips") / "v1_01.jpg").exists()


def test_deliver_respects_the_limit_and_oldest_first(ix):
    assert ib.deliver(ix, limit=1) == (1, 0)
    assert status(ix, "v1_01") == "delivered" and status(ix, "v1_02") == "rendered"


def test_failed_send_leaves_the_clip_rendered(ix):
    ix.api.fail["sendVideo"] = TelegramError("sendVideo: network error (ConnectionError)")
    assert ib.deliver(ix, limit=10) == (0, 2)
    assert status(ix, "v1_01") == "rendered"
    assert (ix.cfg.dir("clips") / "v1_01.mp4").exists()


def test_redelivery_replaces_the_old_message(ix):
    ib.deliver(ix, limit=1)
    db.update_clip(ix.conn, "v1_01", status="rendered")  # re-rendered
    (ix.cfg.dir("clips") / "v1_01.mp4").write_bytes(b"new")
    ib.deliver(ix, limit=1)
    assert ix.api.last("deleteMessage") == {"chat_id": CHAT, "message_id": 101}
    assert db.get_clip(ix.conn, "v1_01")["telegram_message_id"] == 102


def test_oversized_clip_is_reencoded_before_upload(ix, monkeypatch, tmp_path):
    video = ix.cfg.dir("clips") / "v1_01.mp4"
    video.write_bytes(b"x" * 2000)
    ix.cfg.telegram.max_upload_mb = 0.001  # 1000 bytes
    commands = []

    def fake_run(cmd, **kw):
        commands.append(cmd)
        out = tmp_path / "work" / "telegram-v1_01" / "v1_01.mp4"
        out.write_bytes(b"x" * 900)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(ib.proc, "run", fake_run)
    ib.deliver(ix, limit=1)
    cmd = commands[0]
    assert cmd[cmd.index("-c:a") + 1] == "copy" and cmd[cmd.index("-b:v") + 1].endswith("k")
    assert status(ix, "v1_01") == "delivered"


# --- buttons -----------------------------------------------------------------------


def test_done_marks_posted_edits_the_message_and_asks_for_links(ix):
    ib.deliver(ix, limit=1)
    press(ix, "done:v1_01")
    row = db.get_clip(ix.conn, "v1_01")
    assert row["status"] == "posted" and row["done_at"]
    edit = ix.api.last("editMessageCaption")
    assert edit["message_id"] == 101 and edit["reply_markup"] == ib.NO_BUTTONS
    assert "✅ Posted" in edit["caption"] and "submit by" in edit["caption"]
    prompt = ix.api.last("sendMessage")
    assert prompt["reply_markup"]["force_reply"] is True
    assert row["links_prompt_id"] == 102
    assert ix.api.last("answerCallbackQuery")["text"] == "Marked as posted"
    press(ix, "done:v1_01")
    assert ix.api.last("answerCallbackQuery")["text"] == "Already posted"


def test_done_does_not_ask_when_links_were_already_sent(ix):
    ib.deliver(ix, limit=1)
    send_text(ix, "https://youtube.com/shorts/a", reply_to=101)
    press(ix, "done:v1_01")
    assert db.get_clip(ix.conn, "v1_01")["links_prompt_id"] is None
    assert "Paste the post links" not in ix.api.last("sendMessage")["text"]


def test_reject_deletes_the_message_and_local_files(ix):
    ib.deliver(ix, limit=1)
    press(ix, "reject:v1_01")
    assert status(ix, "v1_01") == "rejected"
    assert ix.api.last("deleteMessage")["message_id"] == 101
    assert not any(ix.cfg.dir("clips").glob("v1_01.*"))
    assert (ix.cfg.dir("clips") / "v1_02.jpg").exists()


def test_fit_button_queues_a_rerender_with_the_new_layout(ix):
    ib.deliver(ix, limit=1)
    press(ix, "fit:v1_01")
    row = db.get_clip(ix.conn, "v1_01")
    assert (row["status"], row["layout"]) == ("rerender", "fit")
    layouts = {c.id: c.layout for c in load_clips(ix.cfg.dir("transcripts") / "v1.clips.json")}
    assert layouts == {"v1_01": "fit", "v1_02": "crop"}
    assert ix.api.last("editMessageCaption")["reply_markup"] == ib.NO_BUTTONS


def test_buttons_on_a_clip_that_is_not_waiting_do_nothing(ix):
    press(ix, "fit:v1_01")  # still `rendered`, never delivered
    assert status(ix, "v1_01") == "rendered"
    assert "Cannot change" in ix.api.last("answerCallbackQuery")["text"]
    press(ix, "done:nope_01")
    assert ix.api.last("answerCallbackQuery")["text"] == "Unknown clip"


# --- links -------------------------------------------------------------------------


def test_reply_to_a_clip_records_its_links_by_platform(ix):
    ib.deliver(ix, limit=1)
    send_text(ix, "https://youtube.com/shorts/a https://www.instagram.com/reel/b/", reply_to=101)
    posts = db.posts_for(ix.conn, "v1_01")
    assert [(p["platform"], p["url"]) for p in posts] == [
        ("youtube_shorts", "https://youtube.com/shorts/a"),
        ("instagram_reels", "https://www.instagram.com/reel/b/"),
    ]
    assert "Saved 2 link(s)" in ix.api.last("sendMessage")["text"]
    assert [p["duration_sec"] for p in posts] == [30.0, 30.0]  # clip length, for views by length
    assert status(ix, "v1_01") == "delivered"  # links alone do not mark it done
    send_text(ix, "https://youtube.com/shorts/a", reply_to=101)
    assert len(db.posts_for(ix.conn, "v1_01")) == 2
    assert "already saved" in ix.api.last("sendMessage")["text"]


def test_reply_to_the_links_prompt_and_plain_message_after_done(ix):
    ib.deliver(ix, limit=2)
    press(ix, "done:v1_02")
    prompt_id = db.get_clip(ix.conn, "v1_02")["links_prompt_id"]
    send_text(ix, "https://youtu.be/x", reply_to=prompt_id)
    assert [p["url"] for p in db.posts_for(ix.conn, "v1_02")] == ["https://youtu.be/x"]
    press(ix, "done:v1_01")
    send_text(ix, "https://instagram.com/reel/y")  # not a reply: latest open prompt
    assert [p["url"] for p in db.posts_for(ix.conn, "v1_01")] == ["https://instagram.com/reel/y"]


def test_links_without_a_target_ask_which_clip(ix):
    send_text(ix, "https://youtu.be/x")
    assert "Which clip" in ix.api.last("sendMessage")["text"]
    assert not ix.conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]


# --- chats, commands, polling --------------------------------------------------------


def test_other_chats_are_ignored(ix):
    ib.deliver(ix, limit=1)
    calls = len(ix.api.calls)
    press(ix, "reject:v1_01", chat=666)
    send_text(ix, "/status", chat=666)
    send_text(ix, "https://youtu.be/x", reply_to=101, chat=666)
    assert len(ix.api.calls) == calls  # not even an answer
    assert status(ix, "v1_01") == "delivered"
    assert not db.posts_for(ix.conn, "v1_01")


def test_commands(ix):
    send_text(ix, "/status")
    assert ix.api.last("sendMessage")["text"].startswith("<pre>VIDEOS BY STATUS")
    send_text(ix, "/pending")
    assert ix.api.last("sendMessage")["text"] == "Nothing waiting. 🎉"
    ib.deliver(ix, limit=2)
    send_text(ix, "/pending@clipperdeliverybot")
    assert "2 clip(s) waiting" in ix.api.last("sendMessage")["text"]
    send_text(ix, "/help")
    assert "Reject" in ix.api.last("sendMessage")["text"]


def test_poll_handles_updates_once_and_stores_the_offset(ix):
    ib.deliver(ix, limit=1)
    ix.api.updates = [
        {"update_id": 7, "callback_query": {"id": "a", "data": "done:v1_01",
                                            "message": {"message_id": 101, "chat": {"id": CHAT}}}},
        {"update_id": 8, "message": {"message_id": 5, "chat": {"id": CHAT}, "text": "/help"}},
    ]  # fmt: skip
    assert ib.poll(ix, 0) == 2
    assert status(ix, "v1_01") == "posted"
    assert db.get_setting(ix.conn, ib.OFFSET_KEY) == "9"
    ix.api.updates = []
    ib.poll(ix, 0)
    assert ix.api.last("getUpdates")["offset"] == 9


def test_housekeeping_deletes_posted_messages_after_the_delay(ix):
    ib.deliver(ix, limit=2)
    press(ix, "done:v1_01")
    press(ix, "done:v1_02")
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat(timespec="seconds")
    db.update_clip(ix.conn, "v1_01", done_at=old)
    assert ib.housekeeping(ix) == 1
    assert ix.api.last("deleteMessage")["message_id"] == 101
    assert db.get_clip(ix.conn, "v1_01")["message_deleted"] == 1
    assert ib.housekeeping(ix) == 0  # v1_02 is recent


# --- linking -------------------------------------------------------------------------


def test_find_start_takes_the_latest_private_start():
    updates = [
        {"update_id": 1, "message": {"text": "/start", "chat": {"id": 1, "type": "private"}}},
        {"update_id": 2, "message": {"text": "/start", "chat": {"id": 2, "type": "group"}}},
        {"update_id": 3, "message": {"text": "hello", "chat": {"id": 3, "type": "private"}}},
    ]
    assert ib.find_start(updates)["id"] == 1
    assert ib.find_start([]) is None


def test_write_chat_id_keeps_comments(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "machine:\n  data_dir: ./data\n\n"
        "telegram:                  # inbox\n"
        "  enabled: false           # on/off\n"
        "  chat_id: null            # set by link\n"
        "  max_per_run: 10\n\n"
        "claude:\n  model: sonnet\n"
    )
    ib.write_chat_id(path, 4242)
    text = path.read_text()
    assert "  chat_id: 4242            # set by link\n" in text
    assert "  enabled: true           # on/off\n" in text
    assert "telegram:                  # inbox\n" in text and "claude:\n  model: sonnet\n" in text
    ib.write_chat_id(path, 77)
    assert "  chat_id: 77            # set by link\n" in path.read_text()


def test_write_chat_id_adds_a_block_when_missing(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("machine:\n  data_dir: ./data\n")
    ib.write_chat_id(path, 5)
    assert path.read_text().endswith("\n\ntelegram:\n  enabled: true\n  chat_id: 5\n")


# --- API client ----------------------------------------------------------------------

TOKEN = "123456:SECRET-token-value"


class FakeResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.posts = []

    def post(self, url, data=None, files=None, timeout=None):
        self.posts.append((url, data, files))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(api_mod.time, "sleep", waits.append)
    return waits


def test_api_encodes_fields_and_returns_result(no_sleep):
    session = FakeSession(FakeResponse(200, {"ok": True, "result": {"message_id": 1}}))
    bot = BotAPI(TOKEN, session)
    assert bot.call("sendMessage", {"a": 1, "b": True, "c": {"x": 1}, "d": None}) == {
        "message_id": 1
    }
    url, data, _ = session.posts[0]
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert data == {"a": 1, "b": "true", "c": json.dumps({"x": 1})}


def test_api_retries_rate_limits_after_retry_after(no_sleep):
    session = FakeSession(
        FakeResponse(429, {"ok": False, "error_code": 429, "description": "Too Many Requests",
                           "parameters": {"retry_after": 7}}),
        FakeResponse(200, {"ok": True, "result": True}),
    )  # fmt: skip
    assert BotAPI(TOKEN, session).call("sendMessage") is True
    assert no_sleep == [7.5]


def test_api_does_not_retry_bad_requests(no_sleep):
    session = FakeSession(
        FakeResponse(400, {"ok": False, "error_code": 400, "description": "Bad Request: x"})
    )
    with pytest.raises(TelegramError) as err:
        BotAPI(TOKEN, session).call("sendMessage")
    assert err.value.code == 400 and no_sleep == []


def test_api_network_errors_never_leak_the_token(no_sleep):
    boom = requests.ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/getMe")
    session = FakeSession(boom, boom, boom)
    with pytest.raises(TelegramError) as err:
        BotAPI(TOKEN, session).call("getMe", retries=2)
    assert TOKEN not in str(err.value) and "SECRET" not in str(err.value)
    assert str(err.value) == "getMe: network error (ConnectionError)"
    assert no_sleep == [1.0, 2.0]


def test_api_reopens_files_on_retry(no_sleep, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    session = FakeSession(
        FakeResponse(502, {"ok": False, "error_code": 502, "description": "Bad Gateway"}),
        FakeResponse(200, {"ok": True, "result": True}),
    )
    BotAPI(TOKEN, session).call("sendVideo", files={"video": ("v.mp4", video, "video/mp4")})
    first, second = session.posts[0][2]["video"][1], session.posts[1][2]["video"][1]
    assert first is not second and first.closed and second.closed


def test_token_from_env_wins(monkeypatch):
    monkeypatch.setenv("CLIPPER_TELEGRAM_TOKEN", " 1:abc ")
    assert get_token("does-not-exist") == "1:abc"
    monkeypatch.delenv("CLIPPER_TELEGRAM_TOKEN")
    with pytest.raises(TelegramError, match="no bot token"):
        get_token("clipper-test-no-such-service-xyz")
