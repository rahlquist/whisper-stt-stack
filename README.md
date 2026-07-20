# whisper-stt-stack

A single, fast, self-hosted speech-to-text server on the LAN, built to serve
transcription to multiple clients across the network from one box.

## Goal

Run one STT server on the network instead of transcribing on every device. A
low-power box (`slug`) does the heavy lifting on its GPU; clients just send audio
and get text back. The intended use is push-to-talk voice-to-text from a Linux
client (`voxtype` on `yogaman`), but the server speaks the OpenAI transcription
API, so any OpenAI-compatible client works.

> **Note on the Swedish model:** the Swedish (`kb-whisper-large`) backend is a
> leftover artifact of the initial setup, which began as a Swedish+English
> configuration. It was not a deliberate choice — the primary use is English.
> The SV backend is kept only for completeness; the default route is English.

## What's here

- `whisper_build.md` — full from-scratch build + reproducibility doc (packages,
  models, both whisper.cpp builds, proxy, systemd units, firewall, client setup).
- `whisper_openai_proxy.py` — stdlib-only OpenAI-compatible shim in front of
  whisper.cpp's bare `/inference` endpoint. Exposes `/v1/audio/transcriptions`
  and `/v1/audio/translations`.
- `systemd/` — three units that auto-start the stack on boot:
  - English model on the Radeon iGPU via Vulkan (whisper.cpp-amd) — ~5x faster
    than CPU.
  - Swedish model on CPU.
  - The proxy (the only public face, port 8081).
- `nginx/slug-stt.conf` — optional TLS terminator for browser-mic use (secure
  context required for `getUserMedia`).
- `.gitignore` — excludes the multi-GB model files and build artifacts (downloaded
  / built per the doc, never committed).

## Architecture (one-liner)

```
client (voxtype) --HTTP--> slug:8081 (proxy) --> 127.0.0.1:8083 (EN/Vulkan)
                                                   127.0.0.1:8082 (SV/CPU)
```

## Quick start

See `whisper_build.md` for the complete reproducible build. In short:

1. Build whisper.cpp (CPU, Swedish) and whisper.cpp-amd (Vulkan, English).
2. Drop `whisper_openai_proxy.py`, write the three systemd units, enable them.
3. Open `8081` in the firewall.
4. Point the client at `http://slug:8081` (no `/v1` suffix), model `whisper-1`.

## Notes

- The server cannot press "Enter" in a client — auto-submit is a client feature
  (voxtype `auto_submit` / `smart_auto_submit`). Documented in `whisper_build.md`.
- NPU (XDNA2) on this APU is Windows-only for whisper; the iGPU Vulkan path is the
  Linux accelerator.
