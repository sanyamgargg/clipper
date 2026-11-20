"""Post text: caption + footer + exactly 5 hashtags (campaign required first), YouTube title."""

import pytest

from clipper.config import Campaign, CampaignCaptions, Captions, Config, Rules
from clipper.post_text import clean_topic_tags, hashtags, post_caption, youtube_title
from clipper.stages.pick import PickedClip, resolve_clips
from clipper.timing import Word

DEFAULTS = ["#TJR", "#daytrading", "#trading", "#money", "#mindset"]
CFG = Config(captions=Captions(default_hashtags=DEFAULTS,
                               caption_footer="Follow for daily TJR clips 📈"))  # fmt: skip
TJR = Campaign(id="tjr", name="TJR", rules=Rules(required_hashtags=["#TJR"]))


def test_caption_format():
    assert post_caption(CFG, TJR, "Why TJR quit his job") == (
        "Why TJR quit his job\n\nFollow for daily TJR clips 📈\n\n"
        "#TJR #daytrading #trading #money #mindset"
    )


@pytest.mark.parametrize(
    "topic",
    [[], ["#futurestrading"], ["#futurestrading", "#motivation"], ["#motivation", "#tjr"],
     ["#daytrading"], ["#fyp", "#viral"], ["Bad Tag", "#UPPER", "#ok_tag", "#x"],
     ["#a1", "#b2", "#c3"]],
)  # fmt: skip
@pytest.mark.parametrize(
    "campaign",
    [TJR,
     Campaign(id="x", name="x", rules=Rules(required_hashtags=["#TJR", "#ReachClipping"])),
     Campaign(id="y", name="y", rules=Rules(required_hashtags=["#TJR"]),
              captions=CampaignCaptions(default_hashtags=["#stocks", "#TJR", "#wallstreet",
                                                          "#investing", "#finance"]))],
)  # fmt: skip
def test_always_exactly_five_hashtags_with_tjr_first(topic, campaign):
    tags = hashtags(CFG, campaign, topic)
    assert len(tags) == 5 and tags[0] == "#TJR"
    assert len({t.lower() for t in tags}) == 5  # no duplicates
    last_line = post_caption(CFG, campaign, "Title", topic).splitlines()[-1]
    assert last_line.split() == tags


def test_topic_tags_replace_the_last_defaults():
    assert hashtags(CFG, TJR, ["#futurestrading"]) == [
        "#TJR", "#daytrading", "#trading", "#money", "#futurestrading"]  # fmt: skip
    assert hashtags(CFG, TJR, ["#futurestrading", "#motivation"]) == [
        "#TJR", "#daytrading", "#trading", "#futurestrading", "#motivation"]  # fmt: skip
    assert hashtags(CFG, TJR, ["#daytrading"]) == DEFAULTS  # already there: nothing lost


def test_topic_tags_are_cleaned():
    assert clean_topic_tags(["futuresTrading", "#fyp", "two words", "#ok", "#more"]) == [
        "#futurestrading", "#ok"]  # fmt: skip


def test_campaign_overrides_footer():
    c = Campaign(id="z", name="z", captions=CampaignCaptions(caption_footer=""))
    assert "Follow" not in post_caption(CFG, c, "T")


def test_youtube_title_has_shorts_and_max_100_chars():
    assert youtube_title("Why TJR quit") == "Why TJR quit #shorts"
    long = youtube_title("x" * 300)
    assert len(long) == 100 and long.endswith(" #shorts")


def test_pick_keeps_up_to_two_clean_topic_tags():
    words = [Word(f"w{i}.", i * 0.5, i * 0.5 + 0.4) for i in range(20)]
    pick = PickedClip(start="00:00", end="00:05", hook_text="Setup: punch", title="t", score=0.9,
                      hashtags=["#FuturesTrading", "#viral", "#motivation", "#more"])  # fmt: skip
    (clip,) = resolve_clips([pick], words, "vid", TJR, 2, 20)
    assert clip.hashtags == ["#futurestrading", "#motivation"]
