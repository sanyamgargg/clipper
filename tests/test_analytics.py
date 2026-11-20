"""M5 lite: YouTube id parsing, polling schedule, MVR alerts, /views, earnings, reports.
All HTTP is faked."""

from datetime import UTC, datetime, timedelta

import pytest
import requests

from clipper import db
from clipper.analytics import core, reports
from clipper.analytics.core import (
    earnings_usd,
    fetch_youtube,
    find_post,
    instagram_code,
    is_due,
    parse_count,
    poll_interval,
    poll_youtube,
    report_due,
    tick,
    youtube_id,
)
from clipper.config import Analytics, Campaign, Config, Machine, Rules
from clipper.stages.pick import Clip, save_clips
from clipper.telegram import inbox as ib

NOW = datetime(2026, 11, 20, 6, 0, tzinfo=UTC)  # 11:30 IST
YT = "https://youtube.com/shorts/lsmes4Xg-MQ?si=TtmNHLr66chcxRid"
IG = "https://www.instagram.com/reel/Dfvhxm4x4LE/?igsh=abc"
CAMPAIGN = Campaign(id="tjr", name="TJR", rate_per_1k=0.75, mvr_views=5000, max_payout_usd=750,
                    rules=Rules(required_hashtags=["#TJR"]))  # fmt: skip


@pytest.mark.parametrize(
    ("url", "vid"),
    [
        (YT, "lsmes4Xg-MQ"),
        ("https://www.youtube.com/watch?v=lsmes4Xg-MQ&t=3", "lsmes4Xg-MQ"),
        ("https://youtu.be/lsmes4Xg-MQ?si=x", "lsmes4Xg-MQ"),
        ("https://m.youtube.com/shorts/lsmes4Xg-MQ", "lsmes4Xg-MQ"),
        ("https://youtube.com/@TJRTrades", None),
        ("https://notyoutube.com/watch?v=lsmes4Xg-MQ", None),
        (IG, None),
    ],
)
def test_youtube_id(url, vid):
    assert youtube_id(url) == vid


def test_instagram_code():
    assert instagram_code(IG) == "Dfvhxm4x4LE"
    assert instagram_code("https://instagram.com/p/Cx1/") == "Cx1"
    assert instagram_code(YT) is None


def test_polling_schedule():
    assert poll_interval(timedelta(hours=1)) == timedelta(hours=2)
    assert poll_interval(timedelta(days=3)) == timedelta(hours=12)
    assert poll_interval(timedelta(days=20)) == timedelta(hours=24)
    assert poll_interval(timedelta(days=46)) is None
    posted = NOW - timedelta(hours=5)
    assert is_due(posted, None, NOW)
    assert not is_due(posted, NOW - timedelta(minutes=90), NOW)
    assert is_due(posted, NOW - timedelta(minutes=116), NOW)  # 5 min slack
    assert not is_due(NOW - timedelta(days=50), None, NOW)


def test_earnings_only_over_mvr_and_capped():
    assert earnings_usd(4999, CAMPAIGN) == 0
    assert earnings_usd(5000, CAMPAIGN) == 3.75
    assert earnings_usd(20_000, CAMPAIGN) == 15.0
    assert earnings_usd(10_000_000, CAMPAIGN) == 750
    assert earnings_usd(None, CAMPAIGN) == 0


@pytest.mark.parametrize(
    ("text", "n"),
    [("12345", 12345), ("12,345", 12345), ("12.3k", 12300), ("1.2M", 1_200_000), ("5K", 5000),
     ("abc", None), ("-5", None), ("1.2.3k", None)],
)  # fmt: skip
def test_parse_count(text, n):
    assert parse_count(text) == n


def test_report_due_once_a_day_after_the_time():
    morning = datetime(2026, 11, 23, 9, 5)  # a Monday
    assert report_due(morning, "09:00", None)
    assert not report_due(morning, "09:00", "2026-11-23")
    assert not report_due(morning.replace(hour=8), "09:00", None)
    assert report_due(morning, "09:00", "2026-11-16", weekday=0)
    assert not report_due(morning + timedelta(days=1), "09:00", None, weekday=0)


@pytest.fixture
def env(tmp_path):
    cfg = Config(machine=Machine(data_dir=tmp_path, min_free_gb=0), campaigns=[CAMPAIGN],
                 analytics=Analytics())  # fmt: skip
    conn = db.connect(tmp_path)
    db.add_video(conn, "v1", "tjr", "https://x/v1", "Video", 600)
    clips = [
        Clip("v1_01", "v1", 0, 22, "h", "Short one", "Short one #TJR", "r", 0.9, layout="framed"),
        Clip("v1_02", "v1", 30, 70, "h", "Long story", "Long story #TJR", "r", 0.8, story=True),
    ]
    save_clips(clips, cfg.dir("transcripts") / "v1.clips.json")
    for c in clips:
        db.add_clip(conn, c.id, "v1", c.start_sec, c.end_sec, c.hook_text, c.title, c.reason,
                    c.score, c.layout)  # fmt: skip
    yield cfg, conn
    conn.close()


def _post(conn, clip_id, platform, url, posted_at):
    db.add_post(conn, clip_id, platform, url)
    conn.execute("UPDATE posts SET posted_at = ? WHERE url = ?",
                 (posted_at.isoformat(timespec="seconds"), url))  # fmt: skip
    return conn.execute("SELECT * FROM posts WHERE url = ?", (url,)).fetchone()


class FakeFetch:
    def __init__(self, views):
        self.views = views
        self.calls = []

    def __call__(self, key, ids):
        self.calls.append((key, ids))
        return {i: {"views": self.views, "likes": 3, "comments": 1} for i in ids}


def test_youtube_poll_records_stats_and_fires_mvr_once(env):
    cfg, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW - timedelta(hours=3))
    _post(conn, "v1_01", "instagram_reels", IG, NOW - timedelta(hours=3))  # not polled
    fetch = FakeFetch(6000)
    alerts = poll_youtube(cfg, conn, NOW, key="k", fetch=fetch)
    assert fetch.calls == [("k", ["lsmes4Xg-MQ"])]
    assert [(a.clip_id, a.views, a.mvr, a.title) for a in alerts] == [
        ("v1_01", 6000, 5000, "Short one")
    ]
    post = find_post(conn, YT)
    assert (post["views"], post["platform_video_id"], post["earnings"]) == (
        6000, "lsmes4Xg-MQ", 4.5)  # fmt: skip
    assert post["mvr_reached_at"]
    # not due again for 2 h; when it is, the alert does not repeat
    assert poll_youtube(cfg, conn, NOW + timedelta(minutes=30), key="k", fetch=fetch) == []
    assert len(fetch.calls) == 1
    assert poll_youtube(cfg, conn, NOW + timedelta(hours=2), key="k", fetch=FakeFetch(9000)) == []
    rows = conn.execute("SELECT views, source FROM post_stats ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(6000, "youtube_api"), (9000, "youtube_api")]


def test_poll_without_a_key_does_nothing(env, monkeypatch):
    cfg, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW)
    monkeypatch.setattr(core, "youtube_key", lambda cfg: None)
    assert poll_youtube(cfg, conn, NOW, fetch=FakeFetch(1)) == []


class FakeResp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.params = []

    def get(self, url, params=None, timeout=None):
        self.params.append(params)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_fetch_youtube_batches_and_parses():
    ids = [f"{i:011d}" for i in range(60)]
    body = {"items": [{"id": ids[0], "statistics": {"viewCount": "12", "likeCount": "2"}}]}
    session = FakeSession(FakeResp(200, body), FakeResp(200, {"items": []}))
    out = fetch_youtube("KEY", ids, session, sleep=lambda s: None)
    assert out == {ids[0]: {"views": 12, "likes": 2, "comments": 0}}
    assert [len(p["id"].split(",")) for p in session.params] == [50, 10]


def test_fetch_youtube_errors_never_contain_the_key():
    key = "AIzaSECRETkey123456"
    boom = requests.ConnectionError(f"https://www.googleapis.com/youtube/v3/videos?key={key}")
    waits = []
    with pytest.raises(core.AnalyticsError) as err:
        fetch_youtube(key, ["lsmes4Xg-MQ"], FakeSession(boom, boom, boom), sleep=waits.append)
    assert key not in str(err.value) and waits == [2.0, 4.0]
    denied = FakeResp(403, {"error": {"message": "API key not valid"}})
    with pytest.raises(core.AnalyticsError, match="HTTP 403 API key not valid"):
        fetch_youtube(key, ["x"], FakeSession(denied))


def test_find_post_by_share_link_or_clip_id(env):
    _, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW)
    _post(conn, "v1_01", "instagram_reels", IG, NOW)
    assert find_post(conn, "https://youtu.be/lsmes4Xg-MQ")["platform"] == "youtube_shorts"
    assert (
        find_post(conn, "https://instagram.com/reel/Dfvhxm4x4LE")["platform"] == "instagram_reels"
    )
    assert find_post(conn, "v1_01")["platform"] == "instagram_reels"  # /views is for Instagram
    assert find_post(conn, "v1_02") is None


# --- reports ---------------------------------------------------------------------------


def test_length_buckets():
    assert [reports.length_bucket(s) for s in (18, 25, 35, 35.5, None)] == [
        "<25 s", "25–35 s", "25–35 s", ">35 s", "?"]  # fmt: skip


def test_digest_lists_yesterday_and_instagram_posts_needing_a_count(env):
    cfg, conn = env
    yday = datetime(2026, 11, 19, 14, 0, tzinfo=UTC)  # 19:30 IST yesterday
    yt = _post(conn, "v1_01", "youtube_shorts", YT, yday)
    _post(conn, "v1_02", "instagram_reels", IG, yday)
    core.record_views(cfg, conn, yt, 7000, NOW, "youtube_api")
    text = reports.daily_digest(cfg, conn, NOW)
    assert "Posted yesterday (2)" in text and "Short one · YouTube · 7,000 views" in text
    assert "Estimated earnings (posts over MVR): $5.25" in text
    assert "Instagram views to enter" in text and "Long story (v1_02)" in text


def test_weekly_report_buckets_by_length_story_and_layout(env):
    cfg, conn = env
    a = _post(conn, "v1_01", "youtube_shorts", YT, NOW - timedelta(days=3))
    b = _post(conn, "v1_02", "instagram_reels", IG, NOW - timedelta(days=3))
    core.record_views(cfg, conn, a, 1200, NOW - timedelta(days=2), "youtube_api")  # at 24 h
    core.record_views(cfg, conn, b, 800, NOW - timedelta(days=2), "manual")
    text = reports.weekly_report(cfg, conn, NOW)
    assert "&lt;25 s: median 1,200 (n=1)" in text and "&gt;35 s: median 800 (n=1)" in text
    assert "  story: median 800" in text and "not story: median 1,200" in text
    assert "framed: median 1,200" in text
    assert "Check audience countries" in text
    assert text.index("Short one") < text.index("Long story")  # top first


def test_export_csv(env, tmp_path):
    cfg, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW)
    out = tmp_path / "a" / "clips.csv"
    assert reports.export_csv(cfg, conn, out) == 1
    header, row = out.read_text().splitlines()
    assert header.startswith("post_id,clip_id,platform") and ",YouTube," in row


def test_tick_sends_alerts_and_the_digest_once_per_day(env):
    cfg, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW - timedelta(hours=3))
    sent = []
    morning = datetime(2026, 11, 20, 4, 0, tzinfo=UTC)  # 09:30 IST, a Friday
    assert tick(cfg, conn, sent.append, morning, fetch=FakeFetch(5001), key="k") == 1
    assert sent[0].startswith("🎯 <b>Short one</b> hit 5,000 views on YouTube")
    assert sent[1].startswith("<b>📊 Daily digest")
    assert len(sent) == 2  # not Monday: no weekly report
    tick(cfg, conn, sent.append, morning + timedelta(minutes=10), FakeFetch(5001), "k")
    assert len(sent) == 2


def test_tick_survives_a_failing_api(env):
    cfg, conn = env
    _post(conn, "v1_01", "youtube_shorts", YT, NOW)

    def broken(key, ids):
        raise core.AnalyticsError("YouTube: HTTP 500")

    cfg.analytics.digest_time = "23:59"
    assert tick(cfg, conn, lambda t: None, NOW, fetch=broken, key="k") == 0


# --- Telegram commands -----------------------------------------------------------------


class FakeAPI:
    def __init__(self):
        self.sent = []

    def call(self, method, params=None, files=None, timeout=60, retries=3):
        if method == "sendMessage":
            self.sent.append(params["text"])
        return {"message_id": 1}


def test_views_approved_paid_stats_and_clip_commands(env):
    cfg, conn = env
    _post(conn, "v1_01", "instagram_reels", IG, NOW - timedelta(days=1))
    ix = ib.Inbox(cfg, conn, FakeAPI(), 1)
    ib.on_command(ix, "/views", ["v1_01", "4.2k"])
    assert ix.api.sent[-1] == "Saved 4,200 views for <b>Short one</b> (Instagram)."
    ib.on_command(ix, "/views", ["https://instagram.com/reel/Dfvhxm4x4LE/", "5,100"])
    assert ix.api.sent[-1].startswith("🎯 <b>Short one</b> hit 5,000 views on Instagram")
    ib.on_command(ix, "/views", ["v1_01", "5200"])
    assert not ix.api.sent[-1].startswith("🎯")  # the MVR alert fires once
    ib.on_command(ix, "/approved", [IG])
    ib.on_command(ix, "/paid", [IG, "$3.90"])
    post = find_post(conn, IG)
    assert (post["views"], post["approved"], post["paid_usd"]) == (5200, 1, 3.9)
    ib.on_command(ix, "/views", ["v1_01", "lots"])
    assert ix.api.sent[-1].startswith("Usage: /views")
    ib.on_command(ix, "/views", ["v1_02", "10"])
    assert "No recorded post" in ix.api.sent[-1]
    ib.on_command(ix, "/stats", [])
    assert "Short one · Instagram · 5,200 🎯" in ix.api.sent[-1]
    ib.on_command(ix, "/clip", ["v1_01"])
    assert "Instagram: 5,200 views" in ix.api.sent[-1] and "approved" in ix.api.sent[-1]


def test_commands_with_arguments_are_parsed_from_messages(env):
    cfg, conn = env
    _post(conn, "v1_01", "instagram_reels", IG, NOW)
    ix = ib.Inbox(cfg, conn, FakeAPI(), 1)
    message = {"message_id": 5, "chat": {"id": 1}, "text": "/views@clipperdeliverybot v1_01 900"}
    ib.handle_update(ix, {"update_id": 1, "message": message})
    assert find_post(conn, "v1_01")["views"] == 900
