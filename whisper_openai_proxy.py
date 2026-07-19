#!/usr/bin/env python3
"""
OpenAI-compatible shim for whisper.cpp server.

whisper.cpp (v1.9.1) only exposes POST /inference and GET /health -- there is no
OpenAI-style /v1/audio/transcriptions route (not even in upstream master). This
proxy translates the OpenAI transcription/translation contract onto the backend's
/inference endpoint, so OpenAI-SDK clients work unchanged.

Endpoints:
  POST /v1/audio/transcriptions   -> backend /inference (task=transcribe)
  POST /v1/audio/translations     -> backend /inference (task=translate)
  GET  /v1/models                 -> static model list
  GET  /health                    -> passthrough to backend /health

Env:
  PROXY_HOST      listen host          (default 0.0.0.0)
  PROXY_PORT      listen port          (default 8081)
  BACKEND_URL     default whisper.cpp base url (English model, default http://127.0.0.1:8082)
  BACKEND_SV_URL  Swedish whisper.cpp base url (kb-whisper-large, default http://127.0.0.1:8080)
  Model selection: default = English (medium.en). Send header X-Model: sv to use
  the Swedish kb-whisper-large model. (VoxTalk sends no header -> English.)

  STRIP_NEWLINES  Optional. whisper.cpp inserts "\\n" between segments in its
                  `text` field. When you pause mid-speech, whisper starts a new
                  segment and that newline shows up as a spurious carriage return
                  / line break in the client. Set STRIP_NEWLINES=1 to collapse all
                  internal newlines (\\n, \\r\\n, \\r) to single spaces in the returned
                  `text` (and in verbose_json segment texts). Default OFF (0) so the
                  raw whisper output is passed through unchanged. Set to 1 to enable.

Stdlib only. Python 3.13+ safe (no cgi module used).
"""

import os
import sys
import json
import re
import math
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8082").rstrip("/")  # DEFAULT = English (medium.en)
BACKEND_SV = os.environ.get("BACKEND_SV_URL", "http://127.0.0.1:8080").rstrip("/")  # Swedish (kb-whisper-large)
PROXY_HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8081"))

MODEL_NAME = os.environ.get("MODEL_NAME", "whisper-1")

# Collapse internal newlines (\n, \r\n, \r) to single spaces in returned text.
# Off by default so raw whisper.cpp segment output is preserved. See header.
STRIP_NEWLINES = os.environ.get("STRIP_NEWLINES", "0") == "1"


def backend_for(req_headers):
    """Select backend. Default = English. 'sv'/'swe'/'swedish' (via X-Model header
    or model field) routes to the Swedish kb-whisper-large model."""
    hdr = req_headers.get("X-Model", "").strip().lower()
    return BACKEND_SV if hdr in ("sv", "swe", "swedish") else BACKEND

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _parse_multipart(body: bytes, content_type: str):
    """Return (fields: dict[name->str], file: (filename, bytes) | None)."""
    m = re.search(r'boundary=([^;]+)', content_type)
    if not m:
        raise ValueError("no boundary in Content-Type")
    boundary = m.group(1).strip().strip('"').encode()
    parts = body.split(b"--" + boundary)
    fields = {}
    file_meta = None
    file_bytes = None
    for part in parts:
        if not part or part in (b"", b"--", b"--\r\n"):
            if part.strip() in (b"", b"--"):
                continue
        if part.startswith(b"--"):
            continue
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, raw_content = part.split(b"\r\n\r\n", 1)
        # strip trailing CRLF that precedes the boundary
        if raw_content.endswith(b"\r\n"):
            raw_content = raw_content[:-2]
        header_text = raw_headers.decode("utf-8", "replace")
        disp = re.search(r'name="([^"]+)"', header_text)
        if not disp:
            continue
        name = disp.group(1)
        fname_m = re.search(r'filename="([^"]*)"', header_text)
        if fname_m and fname_m.group(1):
            file_meta = fname_m.group(1)
            file_bytes = raw_content
        else:
            fields[name] = raw_content.decode("utf-8", "replace")
    return fields, (file_meta, file_bytes) if file_meta is not None else None


def _build_multipart(fields: dict, file_info):
    """Rebuild a multipart/form-data body. file_info = (filename, bytes)."""
    boundary = b"----whisperproxyboundary7Qxk"
    body = b""
    for key, val in fields.items():
        body += b"--" + boundary + b"\r\n"
        body += ('Content-Disposition: form-data; name="%s"\r\n\r\n' % key).encode()
        body += val.encode("utf-8") + b"\r\n"
    if file_info:
        fname, fbytes = file_info
        body += b"--" + boundary + b"\r\n"
        body += ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % fname).encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += fbytes + b"\r\n"
    body += b"--" + boundary + b"--\r\n"
    ctype = "multipart/form-data; boundary=" + boundary.decode()
    return body, ctype


def _forward_to_backend(multipart_body, content_type, target=BACKEND):
    req = urllib.request.Request(
        target + "/inference",
        data=multipart_body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _map_response(openai_format, backend_json, duration_sec):
    """Map whisper.cpp /inference JSON to OpenAI shape."""
    try:
        data = json.loads(backend_json)
    except Exception:
        return backend_json  # not JSON; pass through

    # Optional: collapse spurious newlines (whisper inserts \n between segments;
    # pausing mid-speech creates a new segment -> unwanted line break).
    def _norm_text(t):
        if not STRIP_NEWLINES or not isinstance(t, str):
            return t
        return re.sub(r"\s*\n\s*", " ", t).strip()

    if openai_format == "verbose_json":
        segs = data.get("segments", [])
        out_segs = []
        for s in segs:
            out_segs.append({
                "id": s.get("id", 0),
                "seek": 0,
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "text": _norm_text(s.get("text", "")),
                "tokens": s.get("tokens", []),
                "temperature": s.get("temperature", 0.0),
                "avg_logprob": s.get("avg_logprob", 0.0),
                "compression_ratio": 0.0,           # not implemented by backend
                "no_speech_prob": s.get("no_speech_prob", 0.0),
            })
        return json.dumps({
            "task": "transcribe",
            "language": data.get("language", "en"),
            "duration": duration_sec,
            "text": _norm_text(data.get("text", "")),
            "segments": out_segs,
        })
    # json (default) and others: OpenAI json returns {"text": ...}
    if openai_format == "json":
        return json.dumps({"text": _norm_text(data.get("text", ""))})
    # text / srt / vtt: return body as-is
    return backend_json


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "whisper_web.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
            return
        if path in ("/health",):
            try:
                with urllib.request.urlopen(BACKEND + "/health", timeout=10) as r:
                    self._send(r.status, r.read())
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}))
            return
        if path == "/v1/models":
            self._send(200, json.dumps({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": 0,
                    "owned_by": "whisper.cpp",
                    "permission": [],
                    "root": MODEL_NAME,
                    "parent": None,
                }],
            }))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/v1/audio/transcriptions", "/v1/audio/translations"):
            self._send(404, json.dumps({"error": "not found"}))
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        try:
            fields, file_info = _parse_multipart(body, ctype)
        except Exception as e:
            self._send(400, json.dumps({"error": "bad multipart: %s" % e}))
            return

        if not file_info:
            self._send(400, json.dumps({"error": "no 'file' field in request"}))
            return

        # Map OpenAI fields -> whisper.cpp /inference fields
        out_fields = {}
        lang = fields.get("language", "").strip()
        if lang and lang != "auto":
            out_fields["language"] = lang
        elif lang == "auto":
            out_fields["detect_language"] = "true"

        if "prompt" in fields:
            out_fields["prompt"] = fields["prompt"]
        if "temperature" in fields:
            out_fields["temperature"] = fields["temperature"]
        if "translate" in fields and fields["translate"].lower() in ("true", "1", "yes"):
            out_fields["translate"] = "true"
        if path == "/v1/audio/translations":
            out_fields["translate"] = "true"

        # response_format: json (default), text, verbose_json, srt, vtt
        resp_fmt = fields.get("response_format", "json").strip()
        # backend uses 'json' (returns {"text"}), 'text', 'verbose_json', 'srt', 'vtt'
        if resp_fmt == "verbose_json":
            out_fields["response_format"] = "verbose_json"
            out_fields["no_timestamps"] = "false"
        elif resp_fmt in ("srt", "vtt", "text", "json"):
            out_fields["response_format"] = resp_fmt
        else:
            out_fields["response_format"] = "json"

        # estimate duration from WAV header (16-bit PCM, 16kHz mono typical)
        duration_sec = _estimate_duration(file_info[1])

        mp_body, mp_ctype = _build_multipart(out_fields, file_info)
        target = backend_for(self.headers)
        status, resp_text = _forward_to_backend(mp_body, mp_ctype, target)
        sys.stderr.write("[PROXY] %s -> backend status=%d len=%d\n"
                         % (path, status, len(resp_text)))
        sys.stderr.flush()
        if status != 200:
            self._send(status, resp_text if resp_text.startswith("{") else
                       json.dumps({"error": resp_text}))
            return

        out = _map_response(resp_fmt, resp_text, duration_sec)
        ctype_out = "application/json" if resp_fmt in ("json", "verbose_json") else "text/plain"
        self._send(200, out, ctype_out)


def _estimate_duration(audio_bytes):
    """Best-effort duration (seconds) from a WAV/RIFF header. 0.0 if unknown."""
    try:
        if audio_bytes[:4] != b"RIFF":
            return 0.0
        # fmt chunk
        import struct
        pos = 12
        sr = 0
        bps = 0
        channels = 0
        while pos + 8 <= len(audio_bytes):
            cid = audio_bytes[pos:pos + 4]
            sz = struct.unpack("<I", audio_bytes[pos + 4:pos + 8])[0]
            if cid == b"fmt ":
                fmt = audio_bytes[pos + 8:pos + 8 + 16]
                channels = struct.unpack("<H", fmt[2:4])[0]
                sr = struct.unpack("<I", fmt[4:8])[0]
                bps = struct.unpack("<I", fmt[8:12])[0]  # byte rate = channels*sr*bits/8
                break
            pos += 8 + sz
        # data chunk size
        dpos = audio_bytes.find(b"data", 12)
        if dpos != -1 and bps > 0:
            dsz = struct.unpack("<I", audio_bytes[dpos + 4:dpos + 8])[0]
            return round(dsz / bps, 2)
    except Exception:
        pass
    return 0.0


def main():
    srv = ThreadingHTTPServer((PROXY_HOST, PROXY_PORT), Handler)
    print("OpenAI-compatible whisper proxy on http://%s:%d -> %s"
          % (PROXY_HOST, PROXY_PORT, BACKEND), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
