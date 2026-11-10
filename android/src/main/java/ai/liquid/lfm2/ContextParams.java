package ai.liquid.lfm2;

/**
 * Configuration parameters for creating an {@link LFM2Context}.
 */
public class ContextParams {
    /** Maximum sequence length (0 = auto-plan via system RAM / 4096). */
    public int maxSeqLen = 4096;

    /** Number of worker threads (0 = auto-detect CPU cores). */
    public int numThreads = 0;

    /** Attention-sink tokens preserved during KV cache rolling shift (default 64). */
    public int numSinkTokens = 64;

    /** Clamp sequence length to prevent Linux/Android OOM killer (default true). */
    public boolean autoClampRam = true;

    /** Sequential page warmup into physical RAM on load (default false). */
    public boolean warmupMmap = false;

    public ContextParams() {}

    public ContextParams(int maxSeqLen, int numThreads, int numSinkTokens, boolean autoClampRam, boolean warmupMmap) {
        this.maxSeqLen = maxSeqLen;
        this.numThreads = numThreads;
        this.numSinkTokens = numSinkTokens;
        this.autoClampRam = autoClampRam;
        this.warmupMmap = warmupMmap;
    }

    public ContextParams setMaxSeqLen(int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        return this;
    }

    public ContextParams setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        return this;
    }

    public ContextParams setNumSinkTokens(int numSinkTokens) {
        this.numSinkTokens = numSinkTokens;
        return this;
    }

    public ContextParams setAutoClampRam(boolean autoClampRam) {
        this.autoClampRam = autoClampRam;
        return this;
    }

    public ContextParams setWarmupMmap(boolean warmupMmap) {
        this.warmupMmap = warmupMmap;
        return this;
    }
}

