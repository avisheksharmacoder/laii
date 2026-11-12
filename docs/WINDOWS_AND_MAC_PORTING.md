# Windows 11 & macOS Porting Guide

> For the full documentation index, see [docs/README.md](README.md).

**LFM2.c (`laii`)** is currently built and verified as a **Linux (`x86_64`, `aarch64`) + Android (`arm64-v8a`) POSIX** engine. The compute core (AVX2/FMA/F16C/AVX-VNNI and ARM64 NEON/DotProd SIMD kernels, GGUF parser logic, BPE tokenizer, LFM2/LFM2.5/LFM3 forward passes, sampler, grammar PDA, speculative decoding, and `include/lfm_ui.h` widgets) is portable C11, while the OS/runtime wrappers rely on Linux/POSIX primitives (`mmap`, `futex`, `/proc`, `/sys`, `epoll`, `X11/Xft`).

This document details every platform boundary so a developer can port cleanly to **native Windows 11** or **macOS** without guessing.

---

## 1. Current Platform Status

| Platform / Target | Status Today | Notes |
| :--- | :--- | :--- |
| **Linux (`x86_64`, `aarch64`)** | **Production / Verified** | Native CLI, `epoll` HTTP/SSE server, `liblfm2.so`, `bin/lfm2-gui` (X11/Xft), Python SDK, static ELF |
| **Android (`arm64-v8a`)** | **Production / Verified** | 16 KB page-size safe `liblfm2.so`, JNI, Kotlin/Java SDK, Scoped Storage `fd`, `lmkd` page-cache hooks |
| **Windows 11 (WSL2 + WSLg)** | **Works Today (Unmodified)** | Runs CLI, `lfm2-server`, Python SDK, and X11 `bin/lfm2-gui` directly via WSLg |
| **Windows 11 Native (`MSVC` / `MinGW`)** | **Port Required** | Requires Win32 PAL for `mmap`, `futex`/threads, `/proc`, aligned `free`, sockets/server, and Win32 GUI |
| **macOS (`arm64` Apple Silicon / `x86_64`)** | **Port Required (Small Delta)** | POSIX `mmap`/`pthread`/`posix_memalign` work; needs Mach/Darwin replacements for `futex`, `auxv`, `/proc`, `epoll`, `.dylib` |

---

## 2. Already Portable Subsystems

No architectural changes are needed in:
- **Inference & Math Core:** GGUF v3 binary parsing, `Q4_K`/`Q5_K`/`Q6_K`/`IQ4_XS`/`Q4_0`/`Q8_0`/`F16` dot products, ShortConv 3-tap/7-tap state math, GQA / SWA KV-cache, MoE routing, sampling, grammar PDA, and speculative decoding (`conv_trace` rollback).
- **Shared Defaults ([`include/lfm_defaults.h`](../include/lfm_defaults.h)):** Pure C11 defaults (`LFM_DEFAULT_MAX_TOKENS 2000U`, `LFM_DEFAULT_CTX_LEN 10000U`).
- **Immediate-Mode UI Library ([`include/lfm_ui.h`](../include/lfm_ui.h)):** Completely decoupled from X11 behind `LfmUiBackend` function pointers (`draw_rect`, `draw_rounded_rect`, `set_clip`, `measure_text`, `line_height`, `draw_text`, `set_cursor`, clipboard, monotonic clock).
- **Python SDK (`python/lfm2.py`):** Uses standard-library `ctypes`; only needs platform library extension discovery (`lfm2.dll` / `liblfm2.dylib` in addition to `liblfm2.so`).

---

## 3. Windows 11 Native Port (`MSVC` / `clang-cl` / `MinGW-w64`)

### 3.1 Core Engine OS Primitives to Replace

| Subsystem | Current Linux Implementation | File(s) | Windows 11 Replacement |
| :--- | :--- | :--- | :--- |
| **GGUF `mmap`** | `<sys/mman.h>`, `open`, `fstat`, `mmap`, `munmap` | [`tools/lfm2_loader.c`](../tools/lfm2_loader.c), [`tools/lfm2_sys_cache.c`](../tools/lfm2_sys_cache.c) | `CreateFileW`, `GetFileSizeEx`, `CreateFileMappingW`, `MapViewOfFile`, `UnmapViewOfFile` |
| **Page Warmup / Evict** | `madvise(MADV_WILLNEED / MADV_DONTNEED)`, `MADV_HUGEPAGE`, `posix_fadvise` | [`src/lfm2_api.c`](../src/lfm2_api.c), [`tools/lfm2_sys_cache.c`](../tools/lfm2_sys_cache.c) | Guard `#ifdef MADV_*`; optional Win32 `PrefetchVirtualMemory` for warmup, `DiscardVirtualMemory` for evict |
| **Spin-Barrier Sleep** | `<linux/futex.h>` (`SYS_futex`) | [`tools/lfm2_threadpool.c`](../tools/lfm2_threadpool.c) | `WaitOnAddress` + `WakeByAddressAll` (`Synchronization.lib`, native since Windows 8 / 11) |
| **Worker Threads** | `<pthread.h>` (`pthread_create`, `pthread_join`, mutex/cond) | [`tools/lfm2_threadpool.c`](../tools/lfm2_threadpool.c), [`tools/lfm2_gui_worker.h`](../tools/lfm2_gui_worker.h) | Win32 `CreateThread`, `SRWLOCK`, `CONDITION_VARIABLE` (or `winpthreads` under MinGW) |
| **P-Core Discovery** | `/sys/devices/system/cpu/cpuN/...` & `sched_setaffinity` | [`tools/lfm2_threadpool.c`](../tools/lfm2_threadpool.c) | `GetLogicalProcessorInformationEx(RelationProcessorCore, ...)` + `SetThreadAffinityMask` / `EfficiencyClass` (detects Intel P-cores vs E-cores) |
| **RAM / RSS Telemetry** | `/proc/meminfo` & `/proc/self/statm` | [`tools/lfm2_sys_cache.c`](../tools/lfm2_sys_cache.c), [`src/lfm2_api.c`](../src/lfm2_api.c) | `GlobalMemoryStatusEx` (`ullAvailPhys`) and `GetProcessMemoryInfo` (`WorkingSetSize`) |
| **64B Aligned Memory** | `posix_memalign` / `aligned_alloc` + `free()` | [`src/lfm2_api.c`](../src/lfm2_api.c), [`tools/lfm3_engine.c`](../tools/lfm3_engine.c), [`tools/*.c`](../tools) | **Wrap both alloc & free:** `_aligned_malloc(sz, 64)` **must** be freed via `_aligned_free(ptr)` (calling `free()` corrupts CRT heap on Windows) |
| **Monotonic Clock** | `clock_gettime(CLOCK_MONOTONIC, ...)` | [`src/lfm2_api.c`](../src/lfm2_api.c), [`tools/*.c`](../tools) | `QueryPerformanceCounter` / `QueryPerformanceFrequency` |
| **DLL Export Macro** | `__attribute__((visibility("default")))` | [`include/lfm2.h`](../include/lfm2.h) | Mirror [`include/lfm_c_api.h`](../include/lfm_c_api.h#L24): `#ifdef _WIN32` -> `__declspec(dllexport)` when building `lfm2.dll`, `__declspec(dllimport)` for consumers |

### 3.2 CLI, HTTP Server & Desktop GUI on Windows 11
- **CLI ([`tools/lfm2_chat.c`](../tools/lfm2_chat.c)):**
  - Replace `isatty(STDIN_FILENO)` with `_isatty(_fileno(stdin))` and call `SetConsoleOutputCP(CP_UTF8)` at startup so UTF-8 tokens render cleanly in Windows Terminal / `cmd.exe`.
  - Replace two-stage `sigaction(SIGINT, ...)` with `SetConsoleCtrlHandler`.
- **HTTP/SSE Server ([`tools/lfm2_server.c`](../tools/lfm2_server.c)) & GUI SSE Client ([`tools/lfm2_gui_worker.h`](../tools/lfm2_gui_worker.h)):**
  - Replace `<sys/epoll.h>` with **`WSAPoll`** (`<winsock2.h>`, `<ws2tcpip.h>`, link `-lws2_32`, call `WSAStartup` once).
  - Use `ioctlsocket(sock, FIONBIO, &mode)` instead of `fcntl(O_NONBLOCK)`, and `closesocket()` instead of `close()`.
  - Replace `MSG_NOSIGNAL` (non-existent on Windows; `WSAENOTSOCK` / no `SIGPIPE` on Win32) with `0` flags on `send()`.
- **Desktop GUI ([`tools/lfm2_gui.c`](../tools/lfm2_gui.c)):**
  - Keep [`include/lfm_ui.h`](../include/lfm_ui.h) and [`tools/lfm2_gui_worker.h`](../tools/lfm2_gui_worker.h) shared.
  - Write a ~300-line Win32 backend implementing `LfmUiBackend`: `CreateWindowExW` + double-buffered DIB/GDI (`CreateCompatibleDC`, `RoundRect`, `DrawTextW` / `ExtTextOutW` after UTF-8 -> UTF-16 `MultiByteToWideChar`) + `SetClipboardData(CF_UNICODETEXT)` + `PostMessage` (instead of `wake_pipe`) for worker wakeups. Link `-lgdi32 -luser32`.

### 3.3 MSVC / MinGW Compiler Notes
- **`ssize_t` & `strcasecmp`:** MSVC does not define `ssize_t` (`typedef ptrdiff_t lfm_ssize_t;`) or `strcasecmp` (`_stricmp`). MinGW-w64 (`x86_64-w64-mingw32-gcc -std=c11`) provides `<pthread.h>`, `strcasecmp`, and `ssize_t`, making MinGW or `clang-cl` the fastest path to a native `.exe`/`.dll`.
- **AVX2 / FMA Intrinsics:** Both GCC/Clang (`-mavx2 -mfma -mf16c`) and MSVC (`/arch:AVX2`) support all `<immintrin.h>` intrinsics already used in the engine.

---

## 4. macOS Port (`Apple Silicon arm64` & `Intel x86_64`)

macOS shares POSIX foundations with Linux (`mmap`, `pthreads`, `posix_memalign`, `clock_gettime`, `SIGPIPE`, `<dirent.h>`), so **~90% of the core runtime compiles with minimal changes**. The specific deltas are:

### 4.1 Core Engine Deltas

| Subsystem | Current Linux Assumption | File(s) | macOS Replacement |
| :--- | :--- | :--- | :--- |
| **ARM64 HWCAP Header** | Unconditional `#include <sys/auxv.h>` (`getauxval(AT_HWCAP)`) | [`include/lfm2_simd_arm64.h`](../include/lfm2_simd_arm64.h#L15) | Wrap `#if defined(__linux__)`. On `__APPLE__ && __aarch64__`, Apple Silicon always guarantees NEON + DotProd + FP16 (or probe `sysctlbyname("hw.optional.arm.FEAT_DotProd", ...)`) |
| **Futex Idle Wait** | `#include <linux/futex.h>` (`SYS_futex`) | [`tools/lfm2_threadpool.c`](../tools/lfm2_threadpool.c#L12) | `pthread_cond_t` + `pthread_mutex_t` wait/broadcast (or private Darwin `__ulock_wait` / `__ulock_wake`) |
| **Big.LITTLE / P-Core Pinning** | `/sys/devices/system/cpu/` + `sched_setaffinity` | [`tools/lfm2_threadpool.c`](../tools/lfm2_threadpool.c) | `sysctlbyname("hw.perflevel0.physicalcpu", ...)` (queries P-core count directly on M1/M2/M3/M4!) and optional `pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)` |
| **RAM / RSS Telemetry** | `/proc/meminfo` & `/proc/self/statm` | [`tools/lfm2_sys_cache.c`](../tools/lfm2_sys_cache.c), [`src/lfm2_api.c`](../src/lfm2_api.c) | `sysctlbyname("hw.memsize", ...)` / `host_statistics64(HOST_VM_INFO64)` and `task_info(..., MACH_TASK_BASIC_INFO, ...)` |
| **Transparent HugePages / `fadvise`** | `MADV_HUGEPAGE`, `POSIX_FADV_DONTNEED` | [`src/lfm2_api.c`](../src/lfm2_api.c), [`tools/*.c`](../tools) | Already safely `#ifdef MADV_HUGEPAGE` / `#ifdef __linux__` guarded; `MADV_SEQUENTIAL` and `MADV_WILLNEED` are valid on macOS |
| **Socket `SIGPIPE`** | `send(..., MSG_NOSIGNAL)` | [`tools/lfm2_server.c`](../tools/lfm2_server.c) | macOS lacks `MSG_NOSIGNAL`; set `setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one))` and pass `0` flags |
| **HTTP Server Event Loop** | `<sys/epoll.h>` (`epoll_wait`) | [`tools/lfm2_server.c`](../tools/lfm2_server.c#L18) | BSD `<sys/event.h>` (`kqueue` / `kevent`) or portable `poll()` |

### 4.2 Build System (`Makefile`) & Python SDK on macOS
- **Shared Library Naming & Linker Flags:**
  - Replace `.so` with `.dylib` (`lib/liblfm2.dylib`), and replace `-Wl,-soname,liblfm2.so -Wl,-z,max-page-size=16384` with `-Wl,-install_name,@rpath/liblfm2.dylib` and `-Wl,-rpath,@loader_path/../lib`.
  - Apple Silicon binaries run natively with **16 KB virtual pages**, matching our Android 15+ dynamic `sysconf(_SC_PAGESIZE)` alignment work.
- **Compiler Flags:**
  - `HAS_AVX2` probe in [`Makefile`](../Makefile) handles `arm64` cleanly (evaluates to `0`). On `arm64` macOS, Clang supports `-mcpu=apple-m1` or `-march=armv8.2-a+dotprod+fp16`.

### 4.3 Desktop GUI (`bin/lfm2-gui`) on macOS
- **Quick Compatibility Path:** Compiling [`tools/lfm2_gui.c`](../tools/lfm2_gui.c) against **XQuartz** (`-I/opt/X11/include -L/opt/X11/lib -lX11 -lXft`) works out of the box if XQuartz is installed.
- **Native macOS Path:** Implement `LfmUiBackend` (`include/lfm_ui.h`) in a ~350-line Objective-C file (`tools/lfm2_gui_macos.m`) backed by `NSWindow` + `NSView` (`drawRect:` via CoreGraphics `CGContext` + CoreText `CTLineCreateWithAttributedString` + `NSPasteboard` + `CFRunLoopSource` for worker wakeups). Link `-framework Cocoa -framework CoreText`.

---

## 5. Recommended Architecture Refactor Before Starting Either Port

To avoid scattering `#ifdef _WIN32` / `#ifdef __APPLE__` across `src/` and `tools/`:

1. **Create one internal Platform Abstraction Header (`include/lfm_pal.h`):**
   - `lfm_mmap_file` / `lfm_munmap_file` / `lfm_madvise_willneed` / `lfm_madvise_dontneed`
   - `lfm_aligned_alloc(align, size)` / `lfm_aligned_free(ptr)`
   - `lfm_futex_wait(addr, expected)` / `lfm_futex_wake_all(addr)`
   - `lfm_get_avail_ram_bytes()` / `lfm_get_rss_bytes()` / `lfm_detect_p_cores()`
   - `lfm_now_ms()` / `lfm_socket_t` / `lfm_set_nosigpipe(sock)`
2. **Keep [`include/lfm_ui.h`](../include/lfm_ui.h) untouched** and place platform windowing backends side-by-side (`lfm2_gui_x11.c`, `lfm2_gui_win32.c`, `lfm2_gui_macos.m`).
3. **Verify each target with existing parity gates:**
   - Run `test_lfm2_api`, `test_lfm2_server`, `test_lfm2_gui`, and `tests/verify_parity.py` (Tiers 1–4) on Windows and macOS to confirm 0.0-ULP parity between Linux, Windows 11, and Apple Silicon.

