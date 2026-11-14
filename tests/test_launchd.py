"""launchd templates and scripts: valid, pointing at scripts that exist, scheduled right."""

import os
import plistlib
import subprocess

import pytest

from clipper.config import REPO_ROOT
from clipper.proc import lock_held, run_lock

LAUNCHD = REPO_ROOT / "launchd"


def render(label: str) -> dict:
    text = (LAUNCHD / f"{label}.plist").read_text()
    text = text.replace("__REPO__", str(REPO_ROOT)).replace("__LOGS__", "/tmp/clipper-logs")
    assert "__" not in text.replace("__REPO__", "")  # every placeholder filled
    return plistlib.loads(text.encode())


def test_nightly_agent_runs_the_script_at_0130():
    plist = render("com.sanyam.clipper")
    assert plist["Label"] == "com.sanyam.clipper"
    assert plist["StartCalendarInterval"] == {"Hour": 1, "Minute": 30}
    assert plist["RunAtLoad"] is False
    script = plist["ProgramArguments"][-1]
    assert script == str(REPO_ROOT / "scripts" / "run_nightly.sh")
    assert "caffeinate -i" in (REPO_ROOT / "scripts" / "run_nightly.sh").read_text()


def test_bot_agent_is_kept_alive():
    plist = render("com.sanyam.clipper.telegram")
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
    assert plist["ProgramArguments"][-1] == str(REPO_ROOT / "scripts" / "run_bot.sh")


@pytest.mark.parametrize("script", ["run_nightly.sh", "run_bot.sh", "install_launchd.sh"])
def test_scripts_parse(script):
    path = REPO_ROOT / "scripts" / script
    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    assert "ANTHROPIC_API_KEY" not in path.read_text().replace("unset ANTHROPIC_API_KEY", "")


def test_nightly_script_reads_the_token_into_claude_env():
    text = (REPO_ROOT / "scripts" / "run_nightly.sh").read_text()
    assert "security find-generic-password" in text
    assert 'export CLAUDE_CODE_OAUTH_TOKEN="$TOKEN"' in text


def test_lock_held_sees_a_live_holder_only(tmp_path):
    assert not lock_held(tmp_path, "bot.lock")
    with run_lock(tmp_path, "bot.lock"):
        assert lock_held(tmp_path, "bot.lock")
    assert not lock_held(tmp_path, "bot.lock")
    (tmp_path / "bot.lock").write_text("999999999")  # stale: no such process
    assert not lock_held(tmp_path, "bot.lock")
    (tmp_path / "bot.lock").write_text(str(os.getpid()))
    assert lock_held(tmp_path, "bot.lock")
