# Android & Kotlin / Java Integration Guide

This guide describes how to run native on-device LFM2 inference in Android applications using Kotlin or Java, targeting modern ARM64 smartphone SoCs (Qualcomm Snapdragon, Google Tensor, MediaTek Dimensity, Samsung Exynos).

---

## 1. Architecture & Smartphone SoC Optimizations

`LFM2.c` delivers a first-class, zero-dependency C11 inference engine with native JNI bindings for Android ART (API 26+) and standard OpenJDK.

### Key Mobile Capabilities
- **Direct C11 Engine Execution:** Zero JVM overhead or intermediate ONNX/TFLite translation. Pure native memory management.
- **ARM64 NEON & DotProd SIMD:** Hand-tuned ARMv8.2-A vector kernels (`vdotq_s32`, FP16 arithmetic) delivering 100+ tok/s on modern mobile CPU clusters.
- **Android 15+ 16 KB Page Alignment:** Compiled with `-Wl,-z,max-page-size=16384` to guarantee compliance with Google Play's mandatory 16 KB ELF segment alignment starting with Android 15 (API 35).
- **Android Low Memory Killer (`lmkd`) Survival:** Native `releasePageCache()` invokes dual-layer `madvise(MADV_DONTNEED)` and `posix_fadvise(POSIX_FADV_DONTNEED)`, immediately dropping resident memory (RSS) from ~800 MB down to ~20–50 MB when an app goes to the background. Pages fault back into RAM seamlessly on demand.
- **Android Scoped Storage & Content URIs:** Direct descriptor loading via `LFM2Model.loadFromFd(pfd.detachFd())` eliminates any need to duplicate multi-gigabyte GGUF models into private app storage.

---

## 2. Adding LFM2 to Android Studio

### Directory Structure
Copy the following files into your Android Studio project:

```
app/
├── src/
│   └── main/
│       ├── jniLibs/
│       │   └── arm64-v8a/
│       │       └── liblfm2.so                # 16KB page-aligned shared library
│       └── java/
│           └── ai/liquid/lfm2/
│               ├── LFM2Model.java            # Model lifecycle & introspection
│               ├── LFM2Context.java          # Stateful chat & inference session
│               ├── ContextParams.java        # Sequence length & thread configuration
│               ├── SampleParams.java         # Temperature, Top-K, Min-P sampler
│               ├── GenerationChunk.java      # Streamed token chunk & think-phase flag
│               ├── PerfStats.java            # Prefill & decode timing telemetry
│               ├── PoolingType.java          # MEAN and LAST_TOKEN embeddings
│               ├── LFM2Exception.java        # Structured runtime errors
│               └── LFM2Model.kt              # Kotlin DSL & Sequence/Flow helpers
```

### Module `build.gradle.kts`
Ensure your app module filters for `arm64-v8a` and enables 16 KB page-aligned ELF loading:

```kotlin
android {
    defaultConfig {
        ndk {
            abiFilters.addAll(listOf("arm64-v8a"))
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false // Enforces uncompressed 16KB alignment
        }
    }
}
```

---

## 3. Kotlin Usage & Jetpack Compose Integration

### Basic Generation
```kotlin
import ai.liquid.lfm2.*

// 1. Load model via zero-copy mmap
val model = LFM2Model.load("/data/data/com.example.app/files/LFM2.5-1.2B-Instruct-Q4_K_M.gguf")

// 2. Create stateful context with auto thread planning
val context = model.createContext(contextParams {
    maxSeqLen = 2048
    numThreads = 4 // e.g. 4 big cores on big.LITTLE SoC
})

// 3. Set persona
context.setSystemPrompt("You are an on-device AI assistant.")

// 4. Stream tokens
val prompt = "Summarize the key advantages of hybrid architectures."
context.generateSequence(prompt, sampleParams {
    temperature = 0.20f
    topK = 40
}).forEach { chunk ->
    print(chunk.text)
}

// 5. Cleanup when finished
context.close()
model.close()
```

---

### Jetpack Compose `ViewModel` with Coroutines `Flow`

```kotlin
package com.example.app

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.liquid.lfm2.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatViewModel : ViewModel() {

    private var model: LFM2Model? = null
    private var context: LFM2Context? = null

    private val _uiText = MutableStateFlow("")
    val uiText = _uiText.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating = _isGenerating.asStateFlow()

    fun initialize(modelPath: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val m = LFM2Model.load(modelPath)
            val ctx = m.createContext(contextParams {
                maxSeqLen = 2048
                numThreads = 4
                autoClampRam = true
            })
            ctx.setSystemPrompt("You are a helpful mobile assistant.")
            model = m
            context = ctx
        }
    }

    fun sendMessage(userPrompt: String) {
        val ctx = context ?: return
        viewModelScope.launch(Dispatchers.IO) {
            _isGenerating.value = true
            _uiText.value = ""

            ctx.generateSequence(userPrompt, sampleParams {
                temperature = 0.30f
                minP = 0.15f
            }).forEach { chunk ->
                withContext(Dispatchers.Main) {
                    _uiText.value += chunk.text
                }
            }

            _isGenerating.value = false
        }
    }

    // Android Lifecycle hook: Release memory when app enters background
    fun onAppBackgrounded() {
        model?.releasePageCache() // Drops RSS to ~20 MB to survive lmkd!
    }

    // Android Lifecycle hook: Prefault pages in background when app resumes
    fun onAppResumed() {
        viewModelScope.launch(Dispatchers.IO) {
            model?.warmupPageCache()
        }
    }

    override fun onCleared() {
        super.onCleared()
        context?.close()
        model?.close()
    }
}
```

---

## 4. Android Scoped Storage & Content URIs

When allowing users to select GGUF models stored on external storage via the Storage Access Framework (`ActivityResultContracts.OpenDocument`), pass the detached descriptor directly to `LFM2Model.loadFromFd()`:

```kotlin
import android.net.Uri
import android.os.ParcelFileDescriptor
import ai.liquid.lfm2.LFM2Model

fun loadModelFromContentUri(uri: Uri): LFM2Model {
    val pfd: ParcelFileDescriptor = contentResolver.openFileDescriptor(uri, "r")
        ?: throw IllegalStateException("Failed to open Content URI: $uri")

    // detachFd() passes ownership of the underlying OS file descriptor without closing it
    val rawFd = pfd.detachFd()
    val fileSize = pfd.statSize

    // LFM2 native engine duplicates the fd internally via dup()
    return LFM2Model.loadFromFd(rawFd, fileSize, /* warmupPages= */ false)
}
```

---

## 5. Pure Java 8+ Implementation

For pure Java codebases or mixed Java/Kotlin apps:

```java
import ai.liquid.lfm2.*;

public class LFM2Sample {
    public static void runInference(String modelPath) {
        // Load model
        try (LFM2Model model = LFM2Model.load(modelPath)) {
            System.out.println("Model loaded: " + model.getArch() + " with " + model.getNumLayers() + " layers");

            // Configure context
            ContextParams ctxParams = new ContextParams()
                .setMaxSeqLen(2048)
                .setNumThreads(4);

            try (LFM2Context ctx = model.createContext(ctxParams)) {
                ctx.setSystemPrompt("You are a helpful assistant.");

                // Synchronous generation
                String response = ctx.generate("Explain quantum computing in one sentence.");
                System.out.println("Response:\n" + response);

                // Or streaming generation
                ctx.beginChatTurn("List 3 healthy fruits.");
                GenerationChunk chunk;
                while ((chunk = ctx.nextChunk()) != null) {
                    System.out.print(chunk.text);
                }
                System.out.println();

                // Compute L2-normalized embedding
                float[] embd = ctx.embed("Vector similarity query", PoolingType.MEAN);
                System.out.println("Generated " + embd.length + "-dim embedding vector");
            }
        }
    }
}
```

---

## 6. Mobile RAM Optimization Guide

| Event | Recommended Action | Native Hook | Effect |
| :--- | :--- | :--- | :--- |
| **App Pauses (`onPause`)** | Release GGUF mmap page cache | `model.releasePageCache()` | Drops process RSS by up to 90%, preventing Android `lmkd` process kill. |
| **App Resumes (`onResume`)** | Warmup GGUF pages in background | `model.warmupPageCache()` | Pre-faults pages into RAM sequentially for zero first-token lag. |
| **Low Memory Warning (`onTrimMemory`)** | Emergency cache purge | `model.releasePageCache()` | Satisfies `TRIM_MEMORY_RUNNING_CRITICAL` alerts from the OS. |
| **App Closes (`onDestroy` / `onCleared`)** | Free buffers & context | `ctx.close()`, `model.close()` | Unmaps virtual address space and frees threadpool structures. |

---

## 7. Compiling `liblfm2.so` for Android

To cross-compile `lib/android-arm64-v8a/liblfm2.so` manually using the Android NDK:

```bash
make android ANDROID_NDK_HOME=/path/to/android-ndk-r26d ANDROID_API=26
```

Verify 16 KB ELF segment alignment:
```bash
llvm-readelf -l lib/android-arm64-v8a/liblfm2.so | grep "LOAD"
# Expected alignment: 0x4000 (16384 bytes)
```

