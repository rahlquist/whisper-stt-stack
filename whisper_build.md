# Whisper STT Stack — Full Build & Reproducibility Doc

This document lets you rebuild the entire self-hosted speech-to-text stack on a
fresh machine from scratch. It captures the exact state of `slug` (the STT box)
as of 2026-07-18.

Owner/host: `slug` (LAN name, resolves via DNS). Client: `yogaman.home.lan`
runs `voxtype` (the transcription client — NOT "voxtalk"; do not rename).
The proxy URL the client uses is `http://slug:8081/v1` (no `/v1` suffix on the
client endpoint — see Client Setup; the client appends `/v1/audio/transcriptions`).

---

## 0. Hardware & platform

- CPU/APU: AMD Ryzen AI 5 340 (Krackan Point) — has an XDNA2 NPU AND a Radeon 840M iGPU.
- iGPU: Radeon 840M = `gfx1152` (Vulkan 1.4 capable). THIS is what accelerates whisper.
- OS: CachyOS Linux (Arch-based), x86_64.
- RAM: enough for the 2.9 GB Swedish model on CPU + the 1.5 GB English model on the iGPU.

### CRITICAL: NPU does NOT accelerate whisper on Linux
The XDNA2 NPU driver is fully loaded (`amdxdna` module, `/dev/accel0`, `xrt-smi`).
However, `lemonade-sdk/whisper.cpp-amd`'s NPU runner is **Windows-only** (their
README: "NPU — RyzenAI | Ryzen AI 300 series | Windows only"). It will NOT run
whisper on Linux. Do not waste time on the NPU path. The iGPU Vulkan path below
is the working accelerator (~5.3x faster than CPU: 1.47s vs 7.8s per ~10s clip).

---

## 1. Packages (Arch/CachyOS)

```
sudo pacman -S --needed --noconfirm \
  base-devel cmake gcc git \
  vulkan-headers vulkan-icd-loader glslang \
  nginx
```

`vulkan-headers` is REQUIRED to compile `ggml-vulkan` (the shader compiler
`glslc`/`glslangValidator` comes with `glslang`; the loader is `vulkan-icd-loader`).
The iGPU Vulkan device is provided by the `radeon`/RADV Mesa driver (already present
on CachyOS). Verify at runtime with `vulkaninfo --summary` (expect `RADV Krackan`).

---

## 2. Models (download once, reused by both builds)

Models are in ggml format and are loaded directly by whisper.cpp / whisper.cpp-amd.
They are large; download them and keep them — both the CPU build and the Vulkan
build use the same `.bin` files.

```
mkdir -p ~/whisper-models/kb-whisper-large-ggml
cd ~/whisper-models/kb-whisper-large-ggml
```

- English (medium.en, ~1.5 GB):
  HuggingFace `KBLab/whisper-medium-ggml` — `ggml-model-medium.en.bin`
  (place as `~/whisper-models/kb-whisper-large-ggml/ggml-model-medium.en.bin`)
- Swedish (kb-whisper-large, ~2.9 GB):
  HuggingFace `KBLab/kb-whisper-large` — `ggml-model.bin`
  (place as `~/whisper-models/kb-whisper-large-ggml/ggml-model.bin`)

Fetch with `hf` (huggingface-cli) or `curl` from the HF resolve URL. These are the
only model files needed; the directory also holds the whisper.cpp source (section 3).

---

## 3. Build A — original whisper.cpp (Swedish/CPU backend)

This is the CPU build used for the Swedish (SV) model. It stays on CPU because the
iGPU Vulkan build is prioritized for English; SV runs fine on CPU.

```
cd ~/whisper-models/kb-whisper-large-ggml
git clone https://github.com/ggerganov/whisper.cpp whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Binary: `whisper.cpp/build/bin/whisper-server`

---

## 4. Build B — whisper.cpp-amd (English/Vulkan backend)

This fork adds the Vulkan backend. We build with `GGML_VULKAN=ON` to use the iGPU.

```
cd ~/whisper-models
git clone https://github.com/lemonade-sdk/whisper.cpp-amd.git
cd whisper.cpp-amd
cmake -B build -DGGML_VULKAN=ON -DWHISPER_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Pinned commit used on slug: `c7aba15b37105167dafbe63a72951d66a1fd2c5a`
("added NPU runner", 2026-06-09). `--depth 1` is fine if you don't need history.

Verify Vulkan is picked up at configure time: the cmake output must include
`Found Vulkan` / `Including Vulkan backend`. At server startup the log must show
`whisper_backend_init_gpu: device 0: Vulkan0 (type: 2)` and
`whisper_model_load: Vulkan0 total size = ... MB`.

Binary: `build/bin/whisper-server`

> NOTE: this repo has NO published GitHub release binaries (the `/releases/latest`
> endpoint 404s), so a source build is mandatory. The NPU runner is Windows-only
> and is irrelevant on Linux.

---

## 5. OpenAI-compatible proxy

The proxy (`~/whisper-models/whisper_openai_proxy.py`) is stdlib-only Python
(no pip deps). It exposes the OpenAI `/v1/audio/transcriptions` + `/v1/audio/translations`
contract on top of whisper.cpp's bare `/inference` endpoint (whisper.cpp v1.9.1 has
no native OpenAI route). It runs in a venv (`~/whisper-env`, Python 3.14) only to
isolate it; stdlib is all it imports.

Create venv (optional but done on slug):

```
python3 -m venv ~/whisper-env
# nothing to pip install — stdlib only
```

Key proxy behaviors (read the file header for full env list):
- `STRIP_NEWLINES=1` — collapses whisper's inter-segment newlines (whisper inserts
  `\n` between segments; pausing mid-speech makes a new segment → spurious line
  break in the client). ON by default in our unit. Set `0` to preserve raw output.
- `ADD_FINAL_NEWLINE=1` — appends a single trailing `\n` to the returned `text`
  (and to each verbose_json segment text). ON by default. NOTE: this is just a
  newline character in the JSON string; it does NOT make the client press Enter.
  Auto-submit / "hit Enter" is the CLIENT's job (see section 9, voxtype auto_submit).
  Keep ADD_FINAL_NEWLINE if you want a trailing line break in the typed text; it is
  independent of submission.
- Model routing: default = English backend. Header `X-Model: sv` (or `swe`/`swedish`)
  routes to the Swedish backend. `voxtype` sends no header → English.
- The proxy's code-level DEFAULTS (`BACKEND_URL`→8082, `BACKEND_SV_URL`→8080) are
  OVERRIDDEN by the systemd unit's Environment lines (section 6). Do not trust the
  code defaults; the unit is the source of truth.

---

## 6. systemd units (auto-start on boot)

Three units in `/etc/systemd/system/`. They are enabled (`WantedBy=multi-user.target`)
so they start on boot. Ordering: proxy `After=` both servers; servers `After=network-online.target`.

### /etc/systemd/system/whisper-en.service  (English, Vulkan iGPU, port 8083)
```
[Unit]
Description=Whisper EN STT server (Vulkan iGPU, whisper.cpp-amd)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rahlquist
WorkingDirectory=/home/rahlquist/whisper-models/whisper.cpp-amd
ExecStart=/home/rahlquist/whisper-models/whisper.cpp-amd/build/bin/whisper-server --model /home/rahlquist/whisper-models/kb-whisper-large-ggml/ggml-model-medium.en.bin --port 8083 --convert
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### /etc/systemd/system/whisper-sv.service  (Swedish, CPU, port 8082)
```
[Unit]
Description=Whisper SV STT server (kb-whisper-large, CPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rahlquist
WorkingDirectory=/home/rahlquist/whisper-models/kb-whisper-large-ggml/whisper.cpp/whisper.cpp
ExecStart=/home/rahlquist/whisper-models/kb-whisper-large-ggml/whisper.cpp/whisper.cpp/build/bin/whisper-server --model /home/rahlquist/whisper-models/kb-whisper-large-ggml/ggml-model.bin --port 8082 --convert
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### /etc/systemd/system/whisper-proxy.service  (OpenAI proxy, port 8081)
```
[Unit]
Description=Whisper OpenAI-compatible proxy (port 8081)
After=whisper-en.service whisper-sv.service
Wants=whisper-en.service whisper-sv.service

[Service]
Type=simple
User=rahlquist
WorkingDirectory=/home/rahlquist/whisper-models
Environment=BACKEND_URL=http://127.0.0.1:8083
Environment=BACKEND_SV_URL=http://127.0.0.1:8082
Environment=PROXY_PORT=8081
Environment=STRIP_NEWLINES=1
ExecStart=/home/rahlquist/whisper-env/bin/python3 /home/rahlquist/whisper-models/whisper_openai_proxy.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable & start:
```
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-en.service whisper-sv.service whisper-proxy.service
```

`--convert` is required on the servers (they transcode input to the format whisper
expects; without it, non-WAV uploads may be rejected).

---

## 7. Firewall (ufw) — persists across reboot

slug's ufw is `active` with default DENY incoming. Only 22 + 8081 are open.
8080/8082 are intentionally loopback-only (not opened). Enable ufw at boot and
allow the proxy port:

```
sudo systemctl enable ufw
sudo ufw allow 22/tcp
sudo ufw allow 8081/tcp comment 'whisper OpenAI proxy - LAN access'
```

The `allow 8081` rule is saved to `/etc/ufw/user.rules` (survives reboot). Verify:
`curl -s -o /dev/null -w '%{http_code}' http://slug:8081/health` → 200 from the LAN.

> Do NOT open 8080/8082 to the LAN — they are internal backends behind the proxy.

---

## 8. Port map (running state)

| Port  | Service                  | Backend build      | Model                    | Bound to   |
|-------|--------------------------|--------------------|--------------------------|------------|
| 8083  | whisper-en (whisper-server) | whisper.cpp-amd (Vulkan iGPU) | ggml-model-medium.en.bin | 127.0.0.1  |
| 8082  | whisper-sv (whisper-server) | whisper.cpp (CPU)  | ggml-model.bin (kb-whisper-large) | 127.0.0.1  |
| 8081  | whisper-proxy (python3)  | — (shim)           | routes to 8083/8082      | 0.0.0.0    |

The proxy is the ONLY public face. Clients hit `http://slug:8081`.

---

## 9. Client setup (voxtype on yogaman.home.lan)

`voxtype` is the transcription client. Its endpoint config is the #1 past failure
point — read carefully.

- **Endpoint must be `http://slug:8081`** — NO `/v1` suffix.
  `voxtype` itself appends `/v1/audio/transcriptions`, so adding `/v1` here
  double-prefixes to `/v1/v1/...` and the proxy returns 404. This bit us; don't repeat it.
- **Model:** `whisper-1` (the proxy's static model name; not the actual model id).
- **Protocol:** plain HTTP is fine for transcription (audio sent unencrypted — acceptable
  on a trusted LAN). No SSH tunnel required; direct LAN works because ufw allows 8081.
- **Mic recording from the browser demo page** (`http://slug:8081/`) is a SEPARATE issue:
  `getUserMedia` requires a secure context. Plain `http://slug:8081` blocks the mic in
  the browser. Fix = HTTPS in front of 8081 (nginx + self-signed cert, see section 10).
  `voxtype` itself captures mic locally on yogaman and POSTs audio, so it is NOT subject
  to the browser secure-context rule — only the web demo page is.

### voxtype auto-submit ("hit Enter" after transcription)

The server/proxy CANNOT press Enter in the client — it only returns text. Auto-submit
is a voxtype feature. In `~/.config/voxtype/config.toml` (managed by the voxtype-settings
GUI), under `[output]` and `[text]`:

```toml
[output]
auto_submit = true          # voxtype sends a real Enter (via ydotool/wtype virtual keyboard) after typing
shift_enter_newlines = false

[text]
smart_auto_submit = true    # submit on detected sentence end / natural pause, not always
```

After editing, restart the user service: `systemctl --user restart voxtype.service`.
With these on, each transcription auto-advances/submits — no manual Enter. The proxy's
`ADD_FINAL_NEWLINE` toggle is unrelated to this (it only adds a `\n` to the typed string);
leave it or disable it as you prefer.

Verification from yogaman:
```
curl -s -m5 -o /dev/null -w '%{http_code}' http://slug:8081/health   # expect 200
curl -m30 -X POST http://slug:8081/v1/audio/transcriptions \
  -F "file=@/path/to/clip.wav;type=audio/wav" -F "model=whisper-1"   # expect {"text": "..."}
```

---

## 10. Optional: HTTPS for browser-mic (nginx + self-signed)

Only needed if you want the web demo page's mic button to work over the LAN
(browser secure-context requirement). Not required for voxtype.

```
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout /etc/nginx/ssl/slug.key -out /etc/nginx/ssl/slug.crt \
  -days 365 -subj "/CN=slug"
sudo chmod 600 /etc/nginx/ssl/slug.key
```

/etc/nginx/sites-available/slug-stt.conf (symlink into /etc/nginx/sites-enabled/):
```
server {
    listen 8443 ssl;
    listen [::]:8443 ssl;
    server_name slug;

    ssl_certificate     /etc/nginx/ssl/slug.crt;
    ssl_certificate_key /etc/nginx/ssl/slug.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 100m;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Add `include /etc/nginx/sites-enabled/*.conf;` inside the `http { }` block of
`/etc/nginx/nginx.conf` if not present. Then:
```
sudo nginx -t
sudo systemctl enable --now nginx
sudo ufw allow 8443/tcp
```
Browser: open `https://slug:8443/`, accept the self-signed cert warning, mic works.

> On slug this TLS config was partially staged (cert generated, site file written)
> but NOT completed (nginx.conf include + enable + ufw 8443 were not finished). It is
> optional and not part of the core voxtype path.

---

## 11. Operational notes

- Restart just the proxy after editing it:
  `sudo systemctl restart whisper-proxy`
- Watch the proxy log: `journalctl -u whisper-proxy -f`
- Each successful transcription logs: `[PROXY] /v1/audio/transcriptions -> backend status=200 len=N`
- Cold-start verified: after a reboot all three units come back (network-online.target
  + After= ordering) and ufw + the 8081 rule persist. `slug` answering its own LAN IP
  with 200 post-reboot confirms it.
- The CPU English server (old 8080) was retired when Vulkan 8083 replaced it. Do not
  resurrect 8080 unless you want a fallback; the binary still exists at the Build A path.

---

## 12. Repro checklist (fresh box)

1. Install packages (section 1).
2. Download models (section 2) into `~/whisper-models/kb-whisper-large-ggml/`.
3. Build A (section 3) and Build B (section 4).
4. Drop `whisper_openai_proxy.py` into `~/whisper-models/`.
5. Create `~/whisper-env` venv (section 5).
6. Write the three systemd units (section 6), `daemon-reload`, `enable --now`.
7. ufw enable + allow 22 + 8081 (section 7).
8. On yogaman: voxtype endpoint `http://slug:8081`, model `whisper-1` (section 9).
9. Verify: `curl http://slug:8081/health` → 200; a real POST returns text.
