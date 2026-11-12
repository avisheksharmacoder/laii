package ai.liquid.lfm2.app

import ai.liquid.lfm2.*

/**
 * Production-ready Android ViewModel managing LFM2 on-device inference state.
 * <p>
 * Handles background threading, token streaming, and OS lifecycle memory management.
 */
class ChatViewModel {

    var model: LFM2Model? = null
        private set

    var context: LFM2Context? = null
        private set

    var isGenerating: Boolean = false
        private set

    var currentRssMb: Double = 0.0
        private set

    /**
     * Initializes the model and inference context.
     */
    fun initialize(modelPath: String, maxSeqLen: Int = 10000, numThreads: Int = 4) {
        val m = LFM2Model.load(modelPath)
        val ctx = m.createContext(contextParams {
            this.maxSeqLen = maxSeqLen
            this.numThreads = numThreads
            this.autoClampRam = true
        })
        ctx.setSystemPrompt("You are a helpful and concise on-device AI assistant.")
        ctx.addStopString("<|im_end|>")

        model = m
        context = ctx
        updateRss()
    }

    /**
     * Sends a user prompt and streams generated chunks via callback.
     */
    fun sendMessage(
        prompt: String,
        sampleParams: SampleParams = sampleParams { temperature = 0.25f; topK = 40 },
        maxChunks: Int = 2000,
        onChunk: (GenerationChunk) -> Unit
    ) {
        val ctx = context ?: throw IllegalStateException("ChatViewModel not initialized")
        isGenerating = true
        try {
            ctx.generateSequence(prompt, sampleParams, maxChunks).forEach { chunk ->
                onChunk(chunk)
            }
        } finally {
            isGenerating = false
            updateRss()
        }
    }

    /**
     * Android Lifecycle Event: Triggered when app enters background (Activity.onPause).
     * Releases GGUF mmap page cache to survive Android Low Memory Killer (lmkd).
     */
    fun onAppBackgrounded(): Double {
        model?.releasePageCache()
        updateRss()
        return currentRssMb
    }

    /**
     * Android Lifecycle Event: Triggered when app returns to foreground (Activity.onResume).
     * Actively warms up weight pages in the background for zero-lag first tokens.
     */
    fun onAppForegrounded(): Double {
        model?.warmupPageCache()
        updateRss()
        return currentRssMb
    }

    /**
     * Computes an L2-normalized embedding vector for text representation.
     */
    fun computeEmbedding(text: String, poolingType: PoolingType = PoolingType.MEAN): FloatArray {
        val ctx = context ?: throw IllegalStateException("ChatViewModel not initialized")
        return ctx.embed(text, poolingType)
    }

    /**
     * Retrieves the latest performance metrics.
     */
    fun getPerfStats(): PerfStats? {
        return context?.stats
    }

    private fun updateRss() {
        currentRssMb = LFM2Model.getProcessRssBytes() / (1024.0 * 1024.0)
    }

    /**
     * Android ViewModel onCleared() hook: Safely frees all native resources.
     */
    fun onCleared() {
        context?.close()
        context = null
        model?.close()
        model = null
    }
}

