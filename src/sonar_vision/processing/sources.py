"""Bounded incremental capture with an HTTPS relay that pins verified public IPs.

The API supports direct HTTPS files/MJPEG, not pages, HLS or arbitrary protocols.
OpenCV never receives an untrusted remote URL or a remote redirect.
"""

import http.client
import ipaddress
import re
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from urllib.parse import SplitResult, urlsplit
from uuid import uuid4

import cv2
import numpy as np

from sonar_vision.processing.roi import Frame, validate_frame


class SourceError(RuntimeError):
    pass


def validate_url(value: str, allowed_hosts: set[str]) -> tuple[SplitResult, str]:
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "https"
            or not host
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or len(value) > 2048
            or any(ord(c) < 33 for c in value)
            or "\\" in value
            or host not in allowed_hosts
        ):
            raise ValueError
        if not parsed.path.lower().endswith((".mp4", ".webm", ".avi", ".mjpg", ".mjpeg")):
            raise SourceError(
                "Use URL HTTPS direta MP4/WebM/AVI/MJPEG; páginas e HLS não são aceitos"
            )
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = sorted({str(a[4][0]) for a in addresses})
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise ValueError
    except (ValueError, OSError) as exc:
        raise SourceError(
            "Fonte não permitida: configure host público em SONAR_VIDEO_HOSTS"
        ) from exc
    return parsed, ips[0]


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str):
        self.ssl_context = ssl.create_default_context()
        super().__init__(host, timeout=8, context=self.ssl_context)
        self.address = address

    def connect(self) -> None:
        connection = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self.ssl_context.wrap_socket(connection, server_hostname=self.host)
        except BaseException:
            connection.close()
            raise


class PublicVideoRelay:
    def __init__(self, url: str, allowed_hosts: set[str], max_bytes: int = 64 * 1024 * 1024):
        self.parsed, self.address = validate_url(url, allowed_hosts)
        self.max_bytes, self.transferred = max_bytes, 0
        self.lock = threading.Lock()
        self.error: str | None = None
        self.token = f"/{uuid4().hex}/video.mp4"
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass  # Never log signed URLs or tokens.

            def do_GET(self) -> None:
                if self.path != relay.token:
                    self.send_error(404)
                    return
                connection = PinnedHTTPSConnection(relay.parsed.hostname or "", relay.address)
                try:
                    headers = {"User-Agent": "SONAR-VISION/0.1", "Accept": "video/*"}
                    requested_range = self.headers.get("Range")
                    if requested_range:
                        if not re.fullmatch(r"bytes=\d+-\d*", requested_range):
                            self.send_error(400)
                            return
                        headers["Range"] = requested_range
                    path = relay.parsed.path
                    if relay.parsed.query:
                        path += "?" + relay.parsed.query
                    connection.request("GET", path, headers=headers)
                    response = connection.getresponse()
                    content_type = response.getheader("Content-Type", "").split(";")[0].lower()
                    if response.status not in (200, 206) or content_type not in {
                        "video/mp4",
                        "video/webm",
                        "video/x-msvideo",
                        "video/avi",
                        "video/mjpeg",
                        "multipart/x-mixed-replace",
                        "application/octet-stream",
                    }:
                        raise SourceError(
                            "Resposta não é vídeo direto; redirecionamentos são recusados"
                        )
                    self.send_response(response.status)
                    self.send_header("Content-Type", content_type)
                    for name in ("Content-Length", "Content-Range", "Accept-Ranges"):
                        value = response.getheader(name)
                        if value:
                            self.send_header(name, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while chunk := response.read(64 * 1024):
                        with relay.lock:
                            relay.transferred += len(chunk)
                            if relay.transferred > relay.max_bytes:
                                raise SourceError("Limite de transferência de 64 MiB atingido")
                        self.wfile.write(chunk)
                except (OSError, http.client.HTTPException, SourceError):
                    relay.error = "Fonte interrompida, inválida, indisponível ou acima do limite"
                finally:
                    connection.close()
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}{self.token}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


class VideoSource:
    def __init__(
        self,
        source: str,
        allowed_hosts: set[str],
        *,
        local: bool = False,
        max_frames: int = 300,
        max_seconds: float = 60,
        fps: float = 5,
        stop: threading.Event | None = None,
    ):
        if not 1 <= max_frames <= 1200 or not 1 <= max_seconds <= 180 or not 0.5 <= fps <= 15:
            raise ValueError("Limites da fonte inválidos")
        self.source, self.allowed_hosts, self.local = source, allowed_hosts, local
        self.max_frames, self.max_seconds, self.fps = max_frames, max_seconds, fps
        self.stop = stop or threading.Event()
        self.capture: cv2.VideoCapture | None = None
        self.relay: PublicVideoRelay | None = None

    def __enter__(self) -> "VideoSource":
        try:
            if self.local:
                path = Path(self.source).resolve(strict=True)
                if not path.is_file():
                    raise SourceError("Vídeo local não é um arquivo")
                target = str(path)
            else:
                self.relay = PublicVideoRelay(self.source, self.allowed_hosts)
                target = self.relay.start()
            self.capture = cv2.VideoCapture(
                target,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    8000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5000,
                ],
            )
            if not self.capture.isOpened():
                raise SourceError(
                    "Não foi possível abrir o vídeo; verifique formato, host e disponibilidade"
                )
            return self
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> Iterator[tuple[int, float, Frame]]:
        if self.capture is None:
            raise SourceError("Fonte precisa estar aberta")
        start = time.monotonic()
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        stride = max(1, int(source_fps / self.fps)) if 0 < source_fps <= 240 else 1
        count = 0
        while count < self.max_frames and not self.stop.is_set():
            now = time.monotonic()
            if now - start >= self.max_seconds:
                break
            wait_seconds = min(
                max(0, count / self.fps - (now - start)),
                max(0, self.max_seconds - (now - start)),
            )
            if count and self.stop.wait(wait_seconds):
                break
            if time.monotonic() - start >= self.max_seconds:
                break
            for _ in range(stride - 1):
                if not self.capture.grab():
                    break
            success, frame = self.capture.read()
            if not success:
                if not count or (self.relay and self.relay.error):
                    raise SourceError("Vídeo sem frames decodificáveis ou stream interrompido")
                break  # EOF / stream termination; never reconnect across a vehicle track.
            bgr_frame = np.asarray(frame, dtype=np.uint8)
            validate_frame(bgr_frame)
            yield count, time.monotonic() - start, bgr_frame
            count += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.relay is not None:
            self.relay.close()
            self.relay = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
