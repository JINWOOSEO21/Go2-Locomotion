"""헤드리스로 도는 시뮬레이션 화면을 브라우저로 내보내는 MJPEG 스트리머.

왜 이게 필요한가:
  Isaac Sim 5.1 의 WebRTC 라이브스트림(--livestream 2)은 TCP 49100 에 더해 UDP 미디어
  대역까지 열려 있어야 하고, 5.x 부터는 브라우저 클라이언트가 없어져 NVIDIA 데스크톱
  앱을 따로 깔아야 한다. 이 머신은 ufw 가 켜져 있어 22 번 외에는 전부 DROP 되므로
  그 경로가 막힌다.

  MJPEG 는 TCP 한 포트에 multipart/x-mixed-replace 로 JPEG 를 밀어 넣는 방식이라
  SSH 로컬 포워딩(-L) 한 줄이면 방화벽을 전혀 건드리지 않고 브라우저로 볼 수 있다.
  화질/지연은 WebRTC 보다 못하지만 설치가 필요 없다.

프레임 출처는 씬에 이미 붙어 있는 record_camera(TiledCamera) 다. 뷰포트를 긁는 것보다
헤드리스에서 안정적이고, play.py --multicam 이 mp4 를 뽑을 때 쓰는 것과 같은 카메라다.
"""

from __future__ import annotations

import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

_BOUNDARY = "locomotionframe"

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Locomotion GO2</title>
<style>
  html,body{{margin:0;height:100%;background:#111;color:#ddd;
            font-family:ui-monospace,monospace;display:flex;
            flex-direction:column;align-items:center;justify-content:center}}
  img{{max-width:100%;max-height:88vh;image-rendering:auto}}
  p{{font-size:13px;opacity:.7;margin:10px 0 0}}
</style></head>
<body>
  <img src="/stream.mjpg" alt="stream">
  <p>SSH 터미널에서 w/s = 속도, &larr;/&rarr; = 조향, space = 정지, q = 종료</p>
</body></html>
"""


class FrameBuffer:
    """최신 JPEG 한 장만 들고 있는 슬롯.

    시뮬 루프는 쓰기만 하고 절대 블록되지 않는다. 뷰어가 느리면 프레임을 흘려 보낼
    뿐이고, 시뮬레이션 속도는 뷰어에 영향받지 않는다.
    """

    def __init__(self):
        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._seq = 0

    def put(self, jpeg: bytes):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def get_newer_than(self, seq: int, timeout: float = 5.0):
        """seq 보다 새 프레임을 기다렸다가 (jpeg, seq) 를 준다. 없으면 (None, seq)."""
        with self._cond:
            if self._seq <= seq:
                self._cond.wait(timeout)
            if self._jpeg is None or self._seq <= seq:
                return None, seq
            return self._jpeg, self._seq


class _Handler(BaseHTTPRequestHandler):
    buffer: FrameBuffer = None  # 클래스 속성으로 주입한다

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _PAGE.format().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream.mjpg":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()
        seq = 0
        try:
            while True:
                jpeg, seq = self.buffer.get_newer_than(seq)
                if jpeg is None:
                    continue
                self.wfile.write(b"--" + _BOUNDARY.encode() + b"\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            # 뷰어가 탭을 닫은 것뿐이다. 시뮬은 계속 돈다.
            pass

    def log_message(self, fmt, *args):
        # 기본 구현은 매 요청을 stderr 에 찍는데, cbreak 모드인 터미널의
        # 키보드 상태줄과 뒤섞여 읽기 어려워지므로 죽인다.
        pass


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegStreamer:
    """시뮬 루프에서 push() 로 프레임을 밀어 넣으면 브라우저로 나간다."""

    def __init__(self, port: int = 8080, host: str = "127.0.0.1", quality: int = 80, every: int = 2):
        """
        host: 기본값은 루프백이다. SSH 터널로만 노출하면 되므로 굳이 0.0.0.0 에
              열어 두지 않는다(ufw 가 막고 있어 어차피 외부에서 못 붙는다).
        every: N 스텝마다 한 장만 인코딩한다. 시뮬은 50Hz 인데 JPEG 인코딩 + 네트워크가
               그만큼 필요하지 않다. 2 면 25fps.
        """
        self.buffer = FrameBuffer()
        self.quality = int(quality)
        self.every = max(1, int(every))
        self._count = 0
        handler = type("_BoundHandler", (_Handler,), {"buffer": self.buffer})
        self._server = _ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://{host}:{port}/"
        print(f"[mjpeg] 스트리밍 준비 완료 -> {self.url}")

    def push(self, rgb: np.ndarray):
        """RGB uint8 (H, W, 3) 한 장을 인코딩해 최신 프레임으로 올린다."""
        self._count += 1
        if self._count % self.every:
            return
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        # cv2 는 BGR 을 기대한다.
        ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            self.buffer.put(enc.tobytes())

    def close(self):
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
