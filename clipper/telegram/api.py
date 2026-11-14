"""Minimal Telegram Bot API client (long polling, no webhooks) and token lookup.

The token is part of every request URL, so it must never reach a log line or an error
message: errors are built from the method name and Telegram's description only, and
urllib3's request logging is kept quiet.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)  # its debug lines contain the URL

API = "https://api.telegram.org/bot{token}/{method}"
TOKEN_ENV = "CLIPPER_TELEGRAM_TOKEN"


class TelegramError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def get_token(keychain_service: str) -> str:
    """Bot token from env CLIPPER_TELEGRAM_TOKEN, else the macOS Keychain."""
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", keychain_service, "-w"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
    except FileNotFoundError:  # not macOS
        proc = None
    token = proc.stdout.strip() if proc and proc.returncode == 0 else ""
    if not token:
        raise TelegramError(
            f"no bot token: set {TOKEN_ENV} or add it to the Keychain: "
            f"security add-generic-password -s {keychain_service} -a bot -w <token>"
        )
    return token


def _form_value(value: Any) -> Any:
    """Bot API form fields: objects as JSON, booleans as true/false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


class BotAPI:
    def __init__(self, token: str, session: requests.Session | None = None) -> None:
        self._token = token
        self._session = session or requests.Session()

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        files: dict[str, tuple[str, Path, str]] | None = None,
        timeout: float = 60,
        retries: int = 3,
    ) -> Any:
        """POST a Bot API method; returns its `result`. Retries rate limits (429, waiting
        `retry_after`), server errors and network errors; other errors raise at once.
        `files` maps a field to (filename, path, mime type)."""
        data = {k: _form_value(v) for k, v in (params or {}).items() if v is not None}
        url = API.format(token=self._token, method=method)
        for attempt in range(retries + 1):
            handles = []
            try:
                upload = None
                if files:
                    upload = {}
                    for field, (name, path, mime) in files.items():
                        handle = path.open("rb")
                        handles.append(handle)
                        upload[field] = (name, handle, mime)
                resp = self._session.post(url, data=data, files=upload, timeout=timeout)
            except requests.RequestException as exc:
                # str(exc) can contain the URL, i.e. the token: report only the type
                error = TelegramError(f"{method}: network error ({type(exc).__name__})")
                wait = 2.0**attempt
            else:
                try:
                    body = resp.json()
                except ValueError:
                    body = {"ok": False, "error_code": resp.status_code,
                            "description": f"HTTP {resp.status_code}"}  # fmt: skip
                if body.get("ok"):
                    return body.get("result")
                code = body.get("error_code", resp.status_code)
                retry_after = (body.get("parameters") or {}).get("retry_after")
                error = TelegramError(
                    f"{method}: {code} {body.get('description', '')}".strip(), code, retry_after
                )
                if code == 429:
                    wait = float(retry_after or 2.0**attempt) + 0.5
                elif code >= 500:
                    wait = 2.0**attempt
                else:
                    raise error
            finally:
                for handle in handles:
                    handle.close()
            if attempt == retries:
                raise error
            log.warning("%s; retrying in %.0f s", error, wait)
            time.sleep(wait)
        raise AssertionError("unreachable")
