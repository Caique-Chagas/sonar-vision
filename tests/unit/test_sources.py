import socket

import cv2
import numpy as np
import pytest

from sonar_vision.processing.sources import SourceError, VideoSource, validate_url


@pytest.mark.parametrize(
    "url",
    [
        "http://video.example/test.mp4",
        "https://localhost/test.mp4",
        "https://video.example/watch?id=2",
        "file:///tmp/video.mp4",
        "https://user:pass@video.example/a.mp4",
        "https://video.example:444/a.mp4",
        "https://video.example/a.mp4#fragment",
        "https://video.example/a.m3u8",
    ],
)
def test_invalid_sources(url):
    with pytest.raises(SourceError):
        validate_url(url, {"video.example"})


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.2", "169.254.169.254", "::1", "192.168.1.2"]
)
def test_dns_private_addresses_blocked(address, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", (address, 443))])
    with pytest.raises(SourceError):
        validate_url("https://video.example/test.mp4", {"video.example"})


def test_dns_public_and_mixed_rebinding(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )
    assert validate_url("https://video.example/a.mp4", {"video.example"})[1] == "93.184.216.34"
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.1", 443))],
    )
    with pytest.raises(SourceError):
        validate_url("https://video.example/a.mp4", {"video.example"})


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((120, 160, 3), np.uint8))
    writer.release()


def test_local_video_limits_and_release(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)
    source = VideoSource(str(path), set(), local=True, max_frames=2, fps=15)
    with source:
        frames = list(source)
    assert len(frames) == 2 and frames[0][2].shape == (120, 160, 3)
    assert source.capture is None


def test_corrupted_video_releases(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    source = VideoSource(str(path), set(), local=True)
    with pytest.raises(SourceError):
        with source:
            list(source)
    assert source.capture is None


def test_video_cancellation(tmp_path):
    path = tmp_path / "sample.avi"
    make_video(path)
    source = VideoSource(str(path), set(), local=True)
    source.stop.set()
    with source:
        assert list(source) == []
