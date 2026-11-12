package ai.liquid.lfm2

/**
 * Kotlin DSL builder for {@link ContextParams}.
 */
fun contextParams(init: ContextParams.() -> Unit = {}): ContextParams {
    return ContextParams().apply(init)
}

/**
 * Kotlin DSL builder for {@link SampleParams}.
 */
fun sampleParams(init: SampleParams.() -> Unit = {}): SampleParams {
    return SampleParams().apply(init)
}

/**
 * Executes a block of code with an {@link LFM2Context}, automatically closing it upon completion.
 */
inline fun <R> LFM2Model.withContext(
    params: ContextParams = ContextParams(),
    block: (LFM2Context) -> R
): R {
    return createContext(params).use(block)
}

/**
 * Returns a cold Kotlin {@link Sequence} of streamed {@link GenerationChunk} objects.
 * Compatible with pure Kotlin Standard Library (zero runtime dependencies).
 * In Android projects using Kotlin Coroutines, can be converted to Flow via {@code asFlow()}.
 *
 * Example:
 * ```kotlin
 * context.generateSequence("Hello LFM2!").forEach { chunk ->
 *     print(chunk.text)
 * }
 * ```
 */
fun LFM2Context.generateSequence(
    userMessage: String,
    sampleParams: SampleParams = SampleParams(),
    maxChunks: Int = 2000
): Sequence<GenerationChunk> = sequence {
    beginChatTurn(userMessage, sampleParams)
    var emitted = 0
    while (maxChunks <= 0 || emitted < maxChunks) {
        val chunk = nextChunk() ?: break
        yield(chunk)
        emitted++
        if (chunk.isDone) break
    }
}

