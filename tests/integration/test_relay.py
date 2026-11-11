import io
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest

from sonar_vision.processing import sources


@pytest.mark.parametrize(
    "status,content_type,allowed",
    [
        (200, "video/mp4", True),
        (206, "video/webm", True),
        (302, "video/mp4", False),
        (200, "text/html", False),
        (403, "video/mp4", False),
    ],
)
def test_relay_allows_only_video_and_never_follows_redirect(
    monkeypatch, status, content_type, allowed
):
    body = b"fake-video-transport-test"  # This test checks HTTP security, not video decoding.

    class Connection:
        closed = False

        def __init__(self, host, address):
            assert host == "video.example" and address == "93.184.216.34"

        def request(self, method, path, headers):
            assert method == "GET" and path == "/video.mp4"

        def getresponse(self):
            stream = io.BytesIO(body)
            return SimpleNamespace(
                status=status,
                read=stream.read,
                getheader=lambda name, default=None: {
                    "Content-Type": content_type,
                    "Content-Length": str(len(body)),
                }.get(name, default),
            )

        def close(self):
            Connection.closed = True

    monkeypatch.setattr(
        sources,
        "validate_url",
        lambda *a: (urlsplit("https://video.example/video.mp4"), "93.184.216.34"),
    )
    monkeypatch.setattr(sources, "PinnedHTTPSConnection", Connection)
    relay = sources.PublicVideoRelay("https://video.example/video.mp4", {"video.example"})
    url = relay.start()
    try:
        if allowed:
            assert httpx.get(url, trust_env=False).content == body
        else:
            with pytest.raises(httpx.RemoteProtocolError):
                httpx.get(url, trust_env=False)
            assert relay.error is not None
        assert httpx.get(url + "wrong", trust_env=False).status_code == 404
    finally:
        relay.close()
    assert Connection.closed
