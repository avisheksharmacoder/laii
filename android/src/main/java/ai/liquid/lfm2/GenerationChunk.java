package ai.liquid.lfm2;

/**
 * A streamed chunk of generated UTF-8 text from an {@link LFM2Context}.
 */
public class GenerationChunk {
    /** The decoded text segment. */
    public final String text;

    /** True if this chunk was generated inside a reasoning/thinking block (&lt;think&gt;...&lt;/think&gt;). */
    public final boolean isThinkPhase;

    /** True if this chunk marks the completion of the turn. */
    public final boolean isDone;

    public GenerationChunk(String text, boolean isThinkPhase, boolean isDone) {
        this.text = text;
        this.isThinkPhase = isThinkPhase;
        this.isDone = isDone;
    }

    @Override
    public String toString() {
        return text;
    }
}

