package ai.liquid.lfm2;

/**
 * Text embedding pooling strategy for vector representation.
 */
public enum PoolingType {
    /** Causal last token state with L2 normalization (standard for decoder models). */
    LAST_TOKEN(0),

    /** Mean pooled token states across sequence with L2 normalization (dense embeddings). */
    MEAN(1);

    public final int nativeId;

    PoolingType(int nativeId) {
        this.nativeId = nativeId;
    }
}

