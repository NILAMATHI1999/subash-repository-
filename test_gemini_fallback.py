import importlib.util
import io
import json
import os
import urllib.error
from pathlib import Path


SCRIPT = Path(__file__).parent / "ForSocial Robots" / "speech_gemini_respeaker.py"
SPEC = importlib.util.spec_from_file_location("speech_gemini", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def unavailable(request):
    return urllib.error.HTTPError(
        request.full_url,
        503,
        "Unavailable",
        None,
        io.BytesIO(b'{"error":{"message":"high demand"}}'),
    )


def test_temporary_errors_retry_then_use_fallback_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []
    delays = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= 3:
            raise unavailable(request)
        return Response({"candidates": [{"content": {"parts": [{"text": "Fallback works"}]}}]})

    reply = MODULE.generate_gemini_reply(
        "hello",
        urlopen=urlopen,
        sleep=delays.append,
    )

    assert reply == "Fallback works"
    assert delays == [1, 2, 4]
    assert "gemini-3.1-flash-lite" in calls[0]
    assert "gemini-3.5-flash-lite" in calls[-1]


def test_all_temporary_failures_return_spoken_apology(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def urlopen(request, timeout):
        raise unavailable(request)

    reply = MODULE.generate_gemini_reply(
        "hello",
        urlopen=urlopen,
        sleep=lambda _delay: None,
    )

    assert reply == "Sorry, I cannot respond right now. Please try again."
