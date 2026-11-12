package ai.liquid.lfm2;

import java.io.Closeable;
import java.io.File;

/**
 * Handle to an in-memory mapped LFM2 neural network model (dense LFM2 or sparse LFM2-MoE).
 * <p>
 * Supports direct file path loading and Android Scoped Storage / Content URI loading
 * via {@code ParcelFileDescriptor.detachFd()}. Provides mobile OS lifecycle hooks
 * ({@link #releasePageCache()} and {@link #warmupPageCache()}) to survive Android's
 * Low Memory Killer ({@code lmkd}).
 */
public class LFM2Model implements Closeable, AutoCloseable {

    static {
        try {
            System.loadLibrary("lfm2");
        } catch (UnsatisfiedLinkError e) {
            // In standalone desktop testing, users may pass -Djava.library.path=./lib
        }
    }

    private long nativeHandle;
    private final String arch;
    private final String activeIsa;
    private final int numLayers;
    private final int numShortConvLayers;
    private final int numGqaLayers;
    private final int hiddenDim;
    private final int ffnDim;
    private final int vocabSize;
    private final int trainCtx;
    private final int numExperts;
    private final int numExpertsUsed;
    private final long fileSizeBytes;

    private LFM2Model(long handle) {
        if (handle == 0) {
            throw new LFM2Exception(LFM2Exception.ERR_INVALID_ARG, "Failed to load LFM2 model: null handle");
        }
        this.nativeHandle = handle;
        this.arch = nativeGetArch(handle);
        this.activeIsa = nativeGetActiveIsa(handle);
        this.numLayers = nativeGetNumLayers(handle);
        this.numShortConvLayers = nativeGetNumShortConvLayers(handle);
        this.numGqaLayers = nativeGetNumGqaLayers(handle);
        this.hiddenDim = nativeGetHiddenDim(handle);
        this.ffnDim = nativeGetFfnDim(handle);
        this.vocabSize = nativeGetVocabSize(handle);
        this.trainCtx = nativeGetTrainCtx(handle);
        this.numExperts = nativeGetNumExperts(handle);
        this.numExpertsUsed = nativeGetNumExpertsUsed(handle);
        this.fileSizeBytes = nativeGetFileSize(handle);
    }

    /**
     * Loads an LFM2 model from a filesystem path with lazy zero-copy page mapping.
     *
     * @param ggufPath Path to the GGUF model file.
     * @return Initialized {@link LFM2Model} instance.
     */
    public static LFM2Model load(String ggufPath) {
        return load(ggufPath, false);
    }

    /**
     * Loads an LFM2 model from a filesystem path with optional sequential page warmup.
     *
     * @param ggufPath    Path to the GGUF model file.
     * @param warmupPages If true, touches all weight pages sequentially into physical RAM.
     * @return Initialized {@link LFM2Model} instance.
     */
    public static LFM2Model load(String ggufPath, boolean warmupPages) {
        if (ggufPath == null) {
            throw new IllegalArgumentException("ggufPath cannot be null");
        }
        File f = new File(ggufPath);
        if (!f.exists()) {
            throw new LFM2Exception(LFM2Exception.ERR_IO, "Model file not found: " + ggufPath);
        }
        long handle = nativeLoadFromFile(ggufPath, warmupPages);
        if (handle == 0) {
            throw new LFM2Exception(LFM2Exception.ERR_CORRUPT_GGUF, "Failed to map or parse GGUF model: " + ggufPath);
        }
        return new LFM2Model(handle);
    }

    /**
     * Loads an LFM2 model from an open Android file descriptor (e.g. Scoped Storage Content URI).
     * <p>
     * Note: The native engine duplicates the descriptor via {@code dup(fd)}.
     * The caller retains ownership of their own descriptor.
     *
     * @param fd Android OS file descriptor (from {@code parcelFileDescriptor.detachFd()}).
     * @return Initialized {@link LFM2Model} instance.
     */
    public static LFM2Model loadFromFd(int fd) {
        return loadFromFd(fd, 0L, false);
    }

    /**
     * Loads an LFM2 model from an open Android file descriptor with size and warmup controls.
     *
     * @param fd          Android OS file descriptor.
     * @param fileSize    File size in bytes (0 to query automatically via {@code fstat}).
     * @param warmupPages If true, touches all weight pages into physical RAM.
     * @return Initialized {@link LFM2Model} instance.
     */
    public static LFM2Model loadFromFd(int fd, long fileSize, boolean warmupPages) {
        if (fd < 0) {
            throw new IllegalArgumentException("Invalid file descriptor: " + fd);
        }
        long handle = nativeLoadFromFd(fd, fileSize, warmupPages);
        if (handle == 0) {
            throw new LFM2Exception(LFM2Exception.ERR_IO, "Failed to map GGUF model from fd: " + fd);
        }
        return new LFM2Model(handle);
    }

    /**
     * Releases GGUF mmap page cache back to the OS via dual-layer {@code madvise(MADV_DONTNEED)}
     * and {@code posix_fadvise(POSIX_FADV_DONTNEED)}.
     * <p>
     * Recommended for Android {@code Activity.onPause()} or {@code ComponentCallbacks2.onTrimMemory()}
     * to drop process RSS to ~20 MB and survive the Android Low Memory Killer ({@code lmkd}).
     * Weight pages fault back into RAM seamlessly when generation resumes.
     */
    public synchronized void releasePageCache() {
        checkNotClosed();
        nativeReleasePageCache(nativeHandle);
    }

    /**
     * Actively warms GGUF weight pages into physical RAM sequentially via {@code MADV_WILLNEED}
     * and strided page touch (supporting 4 KB, 16 KB and 64 KB OS pages).
     * <p>
     * Ideal for invocation on a background coroutine/thread during Android {@code Activity.onResume()}.
     */
    public synchronized void warmupPageCache() {
        checkNotClosed();
        nativeWarmupPageCache(nativeHandle);
    }

    /**
     * Queries the calling process's current Resident Set Size (RSS) in bytes
     * via {@code /proc/self/statm} (Linux / Android).
     */
    public static long getProcessRssBytes() {
        return nativeGetProcessRss();
    }

    /**
     * Tokenizes raw UTF-8 text into an array of token IDs.
     *
     * @param text Text to tokenize.
     * @return Array of integer token IDs.
     */
    public synchronized int[] tokenize(String text) {
        return tokenize(text, 10000);
    }

    /**
     * Tokenizes raw UTF-8 text into an array of token IDs with an explicit capacity limit.
     *
     * @param text      Text to tokenize.
     * @param maxTokens Maximum number of tokens to return.
     * @return Array of integer token IDs.
     */
    public synchronized int[] tokenize(String text, int maxTokens) {
        checkNotClosed();
        if (text == null) return new int[0];
        int[] result = nativeTokenize(nativeHandle, text, maxTokens);
        if (result == null) {
            throw new LFM2Exception(LFM2Exception.ERR_INVALID_ARG, "Tokenization failed");
        }
        return result;
    }

    /**
     * Creates a new stateful inference context using default parameters.
     */
    public synchronized LFM2Context createContext() {
        return createContext(new ContextParams());
    }

    /**
     * Creates a new stateful inference context with custom parameters.
     *
     * @param params Context configuration parameters.
     * @return Initialized {@link LFM2Context}.
     */
    public synchronized LFM2Context createContext(ContextParams params) {
        checkNotClosed();
        if (params == null) params = new ContextParams();
        long ctxHandle = nativeCreateContext(
            nativeHandle,
            params.maxSeqLen,
            params.numThreads,
            params.numSinkTokens,
            params.autoClampRam,
            params.warmupMmap
        );
        if (ctxHandle == 0) {
            throw new LFM2Exception(LFM2Exception.ERR_OOM, "Failed to allocate LFM2Context");
        }
        return new LFM2Context(this, ctxHandle);
    }

    @Override
    public synchronized void close() {
        if (nativeHandle != 0) {
            nativeFree(nativeHandle);
            nativeHandle = 0;
        }
    }

    public boolean isClosed() {
        return nativeHandle == 0;
    }

    long getNativeHandle() {
        checkNotClosed();
        return nativeHandle;
    }

    private void checkNotClosed() {
        if (nativeHandle == 0) {
            throw new IllegalStateException("LFM2Model has already been closed");
        }
    }

    /* Model Introspection Accessors */
    public String getArch() { return arch; }
    public String getActiveIsa() { return activeIsa; }
    public int getNumLayers() { return numLayers; }
    public int getNumShortConvLayers() { return numShortConvLayers; }
    public int getNumGqaLayers() { return numGqaLayers; }
    public int getHiddenDim() { return hiddenDim; }
    public int getFfnDim() { return ffnDim; }
    public int getVocabSize() { return vocabSize; }
    public int getTrainCtx() { return trainCtx; }
    public int getNumExperts() { return numExperts; }
    public int getNumExpertsUsed() { return numExpertsUsed; }
    public long getFileSizeBytes() { return fileSizeBytes; }

    @Override
    public String toString() {
        return String.format(
            "LFM2Model[arch=%s, ISA=%s, layers=%d (conv=%d, gqa=%d), dim=%d, vocab=%d, experts=%d/%d, size=%.1f MB]",
            arch, activeIsa, numLayers, numShortConvLayers, numGqaLayers,
            hiddenDim, vocabSize, numExpertsUsed, numExperts,
            fileSizeBytes / (1024.0 * 1024.0)
        );
    }

    /* Native Declarations */
    private static native long nativeLoadFromFile(String path, boolean warmup);
    private static native long nativeLoadFromFd(int fd, long fileSize, boolean warmup);
    private static native void nativeFree(long handle);
    private static native void nativeReleasePageCache(long handle);
    private static native void nativeWarmupPageCache(long handle);
    private static native long nativeGetProcessRss();
    private static native String nativeGetArch(long handle);
    private static native String nativeGetActiveIsa(long handle);
    private static native int nativeGetNumLayers(long handle);
    private static native int nativeGetNumShortConvLayers(long handle);
    private static native int nativeGetNumGqaLayers(long handle);
    private static native int nativeGetHiddenDim(long handle);
    private static native int nativeGetFfnDim(long handle);
    private static native int nativeGetVocabSize(long handle);
    private static native int nativeGetTrainCtx(long handle);
    private static native int nativeGetNumExperts(long handle);
    private static native int nativeGetNumExpertsUsed(long handle);
    private static native long nativeGetFileSize(long handle);
    private static native int[] nativeTokenize(long handle, String text, int maxTokens);
    private static native long nativeCreateContext(long handle, int maxSeqLen, int numThreads,
                                                   int sinkTokens, boolean autoClampRam, boolean warmupMmap);
}

