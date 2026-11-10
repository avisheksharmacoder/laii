package ai.liquid.lfm2;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;

/**
 * Stateful multi-turn chat and embedding session for an {@link LFM2Model}.
 * <p>
 * Manages recurrent ShortConv 2-token rolling buffers, GQA KV attention cache,
 * dynamic ChatML state, and speculative token drafting.
 */
public class LFM2Context implements Closeable, AutoCloseable {

    private final LFM2Model model;
    private long nativeHandle;

    /* Reusable primitive buffers for zero-allocation decode loop */
    private final byte[] streamBuf = new byte[1024];
    private final int[] metaBuf = new int[2];
    private final double[] statsBuf = new double[7];

    LFM2Context(LFM2Model model, long nativeHandle) {
        this.model = model;
        this.nativeHandle = nativeHandle;
    }

    /**
     * Sets the ChatML system prompt for subsequent conversation turns.
     *
     * @param systemPrompt Instructions defining model persona and constraints.
     */
    public synchronized void setSystemPrompt(String systemPrompt) {
        setSystemPrompt(systemPrompt, null);
    }

    /**
     * Sets the ChatML system prompt along with an optional JSON tool specification.
     *
     * @param systemPrompt Instructions defining model persona.
     * @param toolsJson    JSON string defining available tool functions.
     */
    public synchronized void setSystemPrompt(String systemPrompt, String toolsJson) {
        checkNotClosed();
        int st = nativeSetSystemPrompt(nativeHandle, systemPrompt, toolsJson);
        if (st != 0) {
            throw new LFM2Exception(st, "Failed to set system prompt");
        }
    }

    /**
     * Appends a multi-token stop string to terminate generation early.
     *
     * @param stopStr Token substring to match and halt generation (e.g. "&lt;|im_end|&gt;").
     */
    public synchronized void addStopString(String stopStr) {
        checkNotClosed();
        if (stopStr == null || stopStr.isEmpty()) return;
        int st = nativeAddStop(nativeHandle, stopStr);
        if (st != 0) {
            throw new LFM2Exception(st, "Failed to add stop string: " + stopStr);
        }
    }

    /**
     * Clears all registered multi-token stop strings.
     */
    public synchronized void clearStops() {
        checkNotClosed();
        nativeClearStops(nativeHandle);
    }

    /**
     * Formats and prefills a ChatML user turn using default sampling parameters.
     *
     * @param userMessage User prompt text.
     */
    public synchronized void beginChatTurn(String userMessage) {
        beginChatTurn(userMessage, new SampleParams());
    }

    /**
     * Formats and prefills a ChatML user turn using custom sampling parameters.
     *
     * @param userMessage  User prompt text.
     * @param sampleParams Sampling configuration.
     */
    public synchronized void beginChatTurn(String userMessage, SampleParams sampleParams) {
        checkNotClosed();
        if (userMessage == null) {
            throw new IllegalArgumentException("userMessage cannot be null");
        }
        if (sampleParams == null) sampleParams = new SampleParams();

        int st = nativeBeginTurn(
            nativeHandle,
            userMessage,
            sampleParams.temperature,
            sampleParams.minP,
            sampleParams.topK,
            sampleParams.repPenalty,
            sampleParams.repWindow,
            sampleParams.seed,
            sampleParams.grammarType,
            sampleParams.maxThinkTokens
        );
        if (st != 0) {
            throw new LFM2Exception(st, "Failed to begin chat turn");
        }
    }

    /**
     * Pulls the next streamed chunk of UTF-8 generated text.
     *
     * @return Next {@link GenerationChunk}, or {@code null} when turn generation completes.
     */
    public synchronized GenerationChunk nextChunk() {
        checkNotClosed();
        int st = nativeNextUtf8(nativeHandle, streamBuf, metaBuf);
        int written = metaBuf[0];
        boolean isThink = (metaBuf[1] != 0);

        if (st < 0) {
            throw new LFM2Exception(st, "Inference step failed");
        }

        if (written > 0) {
            String text = new String(streamBuf, 0, written, StandardCharsets.UTF_8);
            boolean isDone = (st == 1);
            return new GenerationChunk(text, isThink, isDone);
        }

        if (st == 1) {
            return null; /* Generation finished */
        }

        return new GenerationChunk("", isThink, false);
    }

    /**
     * Convenience method to generate the complete response for a user prompt synchronously.
     *
     * @param userMessage User prompt text.
     * @return Full generated text response.
     */
    public synchronized String generate(String userMessage) {
        return generate(userMessage, new SampleParams());
    }

    /**
     * Convenience method to generate the complete response for a user prompt with custom params.
     *
     * @param userMessage  User prompt text.
     * @param sampleParams Sampling parameters.
     * @return Full generated text response.
     */
    public synchronized String generate(String userMessage, SampleParams sampleParams) {
        beginChatTurn(userMessage, sampleParams);
        StringBuilder sb = new StringBuilder();
        GenerationChunk chunk;
        while ((chunk = nextChunk()) != null) {
            sb.append(chunk.text);
        }
        return sb.toString();
    }

    /**
     * Computes an L2-normalized embedding vector using the default {@link PoolingType#MEAN} strategy.
     *
     * @param text Input text to embed.
     * @return Float array of length {@code model.getHiddenDim()}.
     */
    public synchronized float[] embed(String text) {
        return embed(text, PoolingType.MEAN);
    }

    /**
     * Computes an L2-normalized embedding vector using a specified pooling strategy.
     *
     * @param text        Input text to embed.
     * @param poolingType Pooling strategy (e.g. {@link PoolingType#MEAN} or {@link PoolingType#LAST_TOKEN}).
     * @return Float array of length {@code model.getHiddenDim()}.
     */
    public synchronized float[] embed(String text, PoolingType poolingType) {
        checkNotClosed();
        if (text == null) {
            throw new IllegalArgumentException("text cannot be null");
        }
        if (poolingType == null) poolingType = PoolingType.MEAN;

        float[] embd = nativeEmbed(nativeHandle, text, poolingType.nativeId);
        if (embd == null) {
            throw new LFM2Exception(LFM2Exception.ERR_INVALID_ARG, "Failed to compute embedding");
        }
        return embd;
    }

    /**
     * Retrieves the current inference performance statistics.
     */
    public synchronized PerfStats getStats() {
        checkNotClosed();
        nativeGetStats(nativeHandle, statsBuf);
        return new PerfStats(
            (int)statsBuf[0],
            (int)statsBuf[1],
            (int)statsBuf[2],
            statsBuf[3],
            statsBuf[4],
            statsBuf[5],
            statsBuf[6]
        );
    }

    /**
     * Returns the current token position cursor in the KV / ShortConv context.
     */
    public synchronized int getPos() {
        checkNotClosed();
        return nativeGetPos(nativeHandle);
    }

    /**
     * Resets the context sequence position and KV cache to position 0 for a new conversation.
     */
    public synchronized void reset() {
        checkNotClosed();
        nativeReset(nativeHandle);
    }

    /**
     * Serializes the current conversation prefix state to a {@code .lfmcache} disk file.
     *
     * @param cacheFilePath Path to write the serialized cache.
     */
    public synchronized void savePromptCache(String cacheFilePath) {
        checkNotClosed();
        if (cacheFilePath == null) throw new IllegalArgumentException("cacheFilePath cannot be null");
        int st = nativeSavePromptCache(nativeHandle, cacheFilePath);
        if (st != 0) {
            throw new LFM2Exception(st, "Failed to save prompt cache: " + cacheFilePath);
        }
    }

    /**
     * Restores conversation state from a {@code .lfmcache} disk file if the prompt prefix matches.
     *
     * @param cacheFilePath Path to the serialized cache file.
     * @param promptTokens  Tokens of the prefix to verify and restore.
     * @return Number of matched tokens restored into the context.
     */
    public synchronized int loadPromptCache(String cacheFilePath, int[] promptTokens) {
        checkNotClosed();
        if (cacheFilePath == null || promptTokens == null) {
            throw new IllegalArgumentException("Arguments cannot be null");
        }
        int[] outMatched = new int[1];
        int st = nativeLoadPromptCache(nativeHandle, cacheFilePath, promptTokens, outMatched);
        if (st != 0) {
            throw new LFM2Exception(st, "Failed to load prompt cache: " + cacheFilePath);
        }
        return outMatched[0];
    }

    /**
     * Returns the parent {@link LFM2Model} associated with this context.
     */
    public LFM2Model getModel() {
        return model;
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

    private void checkNotClosed() {
        if (nativeHandle == 0) {
            throw new IllegalStateException("LFM2Context has already been closed");
        }
        if (model.isClosed()) {
            throw new IllegalStateException("Parent LFM2Model has already been closed");
        }
    }

    /* Native Declarations */
    private static native void nativeFree(long handle);
    private static native void nativeReset(long handle);
    private static native int nativeGetPos(long handle);
    private static native int nativeSetSystemPrompt(long handle, String prompt, String toolsJson);
    private static native void nativeClearStops(long handle);
    private static native int nativeAddStop(long handle, String stopStr);
    private static native int nativeBeginTurn(long handle, String message, float temperature,
                                              float minP, int topK, float repPenalty,
                                              int repWindow, long seed, int grammarType,
                                              int maxThinkTokens);
    private static native int nativeNextUtf8(long handle, byte[] outBuf, int[] outMeta);
    private static native float[] nativeEmbed(long handle, String text, int poolingType);
    private static native void nativeGetStats(long handle, double[] outStats);
    private static native int nativeSavePromptCache(long handle, String path);
    private static native int nativeLoadPromptCache(long handle, String path, int[] tokens, int[] outMatched);
}

