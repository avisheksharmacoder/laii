package ai.liquid.lfm2;

/**
 * Autoregressive sampling parameters for token generation.
 */
public class SampleParams {
    /** Temperature (0.0f = greedy argmax, default 0.30f). */
    public float temperature = 0.30f;

    /** Min-P dynamic truncation threshold (default 0.15f). */
    public float minP = 0.15f;

    /** Top-K candidate pool size (default 50). */
    public int topK = 50;

    /** Repetition penalty (1.0f = disabled, default 1.05f). */
    public float repPenalty = 1.05f;

    /** Repetition penalty lookback ring window size (default 64). */
    public int repWindow = 64;

    /** Random number generator seed (0 = random hardware clock seed). */
    public long seed = 0L;

    /** Constrained grammar type (0 = none, 1 = JSON object, 2 = tool call). */
    public int grammarType = 0;

    /** Maximum allowed thinking tokens before forcing closure (0 = unlimited). */
    public int maxThinkTokens = 0;

    public SampleParams() {}

    public SampleParams setTemperature(float temperature) {
        this.temperature = temperature;
        return this;
    }

    public SampleParams setMinP(float minP) {
        this.minP = minP;
        return this;
    }

    public SampleParams setTopK(int topK) {
        this.topK = topK;
        return this;
    }

    public SampleParams setRepPenalty(float repPenalty) {
        this.repPenalty = repPenalty;
        return this;
    }

    public SampleParams setRepWindow(int repWindow) {
        this.repWindow = repWindow;
        return this;
    }

    public SampleParams setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public SampleParams setGrammarType(int grammarType) {
        this.grammarType = grammarType;
        return this;
    }

    public SampleParams setMaxThinkTokens(int maxThinkTokens) {
        this.maxThinkTokens = maxThinkTokens;
        return this;
    }
}

