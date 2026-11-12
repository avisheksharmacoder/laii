# LFM Studio — Native C11 Desktop GUI (`bin/lfm2-gui`)

> For the full documentation index, see [docs/README.md](README.md).

**LFM Studio (`bin/lfm2-gui`)** is a lightweight, zero-toolkit native desktop chat interface written in pure **C11 + X11 + Xft**. Compiled directly from source in this repository with no Electron, GTK, Qt, or external UI downloads, the binary measures **~97 KB** on `x86_64` and compiles cleanly after `git clone`.

---

## 1. Architecture & Footprint

The GUI is intentionally split into three self-contained layers:

| File | Responsibility |
| :--- | :--- |
| [`include/lfm_ui.h`](../include/lfm_ui.h) | Single-header C11 immediate-mode GUI library (`LFM_UI_IMPLEMENTATION`): Dark/Light palettes, layout slicing (`lfm_rect_cut_*`), clip stack, `LfmUiScroll` (mouse wheel + scrollbar thumb + bottom-stickiness), markdown-lite fenced code card renderer, buttons/chips/toggles/sliders, and multi-line UTF-8 `LfmUiTextEdit` with clipboard selection |
| [`tools/lfm2_gui_worker.h`](../tools/lfm2_gui_worker.h) | Headless worker `pthread` (`LfmGuiEngine`), `.gguf` directory scanner (`lfm_gui_scan_dir`), shared defaults (`include/lfm_defaults.h`: `ctx = 10000`, `max_tokens = 2000`), `liblfm2.so` pull-streaming loop (`lfm2_context_next_utf8`), and pure POSIX TCP/SSE client (`POST /v1/chat/completions`) with split-chunk `<think>...</think>` routing |
| [`tools/lfm2_gui.c`](../tools/lfm2_gui.c) | Native Linux X11 + Xft double-buffered frontend (`XCreatePixmap` + `XftDraw`), `XIM`/`Xutf8LookupString` UTF-8 input, `XA_CLIPBOARD` copy/paste, and `poll()` event loop over `ConnectionNumber(dpy)` + worker wake pipe (`0%` idle CPU) |

---

## 2. Building (`make gui`)

On systems with X11 and Xft development headers installed, `make all` automatically builds `bin/lfm2-gui`. You can also build it directly:

```bash
make gui
```

### Linux System Packages (Fresh Machine)
* **Ubuntu / Debian:** `sudo apt install libx11-dev libxft-dev libfontconfig1-dev`
* **Fedora / RHEL:** `sudo dnf install libX11-devel libXft-devel fontconfig-devel`
* **Arch Linux:** `sudo pacman -S libx11 libxft fontconfig`

> Headless servers and CI runners without `x11`/`xft` `pkg-config` packages still succeed under `make all` (the `gui` target is skipped automatically when `pkg-config --exists x11 xft` is false).

---

## 3. Running `bin/lfm2-gui`

### Local In-Process Inference (`liblfm2.so`)
```bash
# Auto-detect models/LFM3-1.7B-Q4_K_M.gguf (or first sorted .gguf in ./models)
./bin/lfm2-gui

# Launch with an explicit checkpoint and Light theme
./bin/lfm2-gui -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf --light

# Customize context window, token cap (0 = unlimited), and threads
./bin/lfm2-gui -m models/LFM3-1.7B-Q4_K_M.gguf --ctx 10000 -n 2000 -j 4
```

### Remote / Server Mode (`bin/lfm2-server`)
Switch between **Local** and **Server** in the top header bar, or start directly in Server mode against a running [`bin/lfm2-server`](../tools/lfm2_server.c):

```bash
# Terminal 1: Start inference server
./bin/lfm2-server -m models/LFM3-1.7B-Q4_K_M.gguf --port 8080

# Terminal 2: Launch desktop GUI client
./bin/lfm2-gui --server 127.0.0.1:8080
```

---

## 4. CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-m`, `--model <path.gguf>` | `models/LFM3-1.7B-Q4_K_M.gguf` (auto) | Path to local `.gguf` checkpoint |
| `--ctx <tokens>` | `10000` | KV/context window size (`LFM_DEFAULT_CTX_LEN`) |
| `-n`, `--max-tokens <tokens>` | `2000` | Max generated tokens per turn (`0` = unlimited) |
| `-j`, `--threads <n>` | `0` (`auto`) | Worker CPU threads |
| `--server <host[:port]>` | `127.0.0.1:8080` | Enable remote `lfm2-server` HTTP/SSE backend |
| `--warmup` | off | Enable sequential `mmap` page warmup on load |
| `--light` | off (Dark default) | Start with the Light color palette |
| `--smoke-frames <n>` | `0` (interactive) | Render `n` X11 frames cleanly and exit (CI verification) |
| `-h`, `--help` | — | Print CLI usage summary |

---

## 5. Desktop Controls & Features

* **Header Bar:** Model browser trigger (`.gguf` file & directory picker modal), backend toggle (`Mode: Local` / `Mode: Server`), live status indicator, `New Chat` (resets KV/ShortConv state), `Light`/`Dark` theme toggle, and `Settings` drawer toggle.
* **Composer:** `Enter` sends the prompt; `Shift+Enter` inserts a newline; `Ctrl+A` / `Ctrl+C` / `Ctrl+X` / `Ctrl+V` work against system `XA_CLIPBOARD`; clicking `Stop` cancels active generation safely without corrupting session state.
* **Chat Cards:** Collapsible `<think>` reasoning panels (auto-opened during thinking streams), fenced code cards (` ```lang ... ``` `), per-response `Copy` buttons, token counts, decode `tok/s`, and truncation/stop notices (`length`, `cancelled`, `context_full`).
* **Settings Drawer:** Live sliders/chips for `temperature`, `min_p`, `top_k`, `max_tokens` (`512` / `2000` / `4096` / `∞`), context window (`2k` / `10k` / `32k`), thread count, `mmap` warmup, and system prompt presets.
* **Telemetry Strip:** Live `ctx: pos / max`, `prefill ms`, `decode tok/s`, and process `RSS MB`.

---

## 6. Verification

* **Headless C Unit & Worker Integration Tests (`bin/test_lfm2_gui`, included in `make test-c-bins`):**
  ```bash
  ./bin/test_lfm2_gui
  ```
  Verifies layout geometry, UTF-8 wrapping, markdown code-block sizing, multi-line `LfmUiTextEdit` clipboard/editing, SSE JSON delta parsing, split-chunk `<think>` routing, `.gguf` directory discovery, and real worker `pthread` inference against `tests/lfm2_toy.gguf` without requiring an X11 server.
* **Live X11 Render Smoke Check:**
  ```bash
  DISPLAY=:0 ./bin/lfm2-gui --smoke-frames 5 -m tests/lfm2_toy.gguf
  ```

