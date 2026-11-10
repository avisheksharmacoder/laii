package ai.liquid.lfm2;

/**
 * Performance metrics and generation counters for an {@link LFM2Context}.
 */
public class PerfStats {
    public final int promptTokens;
    public final int generatedTokens;
    public final int currentPos;
    public final double prefillMs;
    public final double decodeMs;
    public final double prefillTokPerSec;
    public final double decodeTokPerSec;

    public PerfStats(int promptTokens, int generatedTokens, int currentPos,
                     double prefillMs, double decodeMs,
                     double prefillTokPerSec, double decodeTokPerSec) {
        this.promptTokens = promptTokens;
        this.generatedTokens = generatedTokens;
        this.currentPos = currentPos;
        this.prefillMs = prefillMs;
        this.decodeMs = decodeMs;
        this.prefillTokPerSec = prefillTokPerSec;
        this.decodeTokPerSec = decodeTokPerSec;
    }

    @Override
    public String toString() {
        return String.format(
            "PerfStats[prompt=%d tok (%.1f ms, %.1f tok/s), gen=%d tok (%.1f ms, %.1f tok/s), pos=%d]",
            promptTokens, prefillMs, prefillTokPerSec,
            generatedTokens, decodeMs, decodeTokPerSec, currentPos
        );
    }
}

