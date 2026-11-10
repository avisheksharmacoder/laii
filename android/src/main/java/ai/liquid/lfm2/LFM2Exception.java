package ai.liquid.lfm2;

/**
 * Exception thrown when a native LFM2 engine operation encounters an error.
 */
public class LFM2Exception extends RuntimeException {
    public static final int ERR_INVALID_ARG  = -1;
    public static final int ERR_IO           = -2;
    public static final int ERR_CORRUPT_GGUF = -3;
    public static final int ERR_OOM          = -4;
    public static final int ERR_CTX_FULL     = -5;
    public static final int ERR_CHECKPOINT   = -6;

    public final int errorCode;

    public LFM2Exception(int errorCode, String message) {
        super(resolveMessage(errorCode, message));
        this.errorCode = errorCode;
    }

    private static String resolveMessage(int code, String message) {
        String base = (message != null) ? message : "LFM2 engine error";
        switch (code) {
            case ERR_INVALID_ARG:
                return base + " [ERR_INVALID_ARG: invalid parameter or null handle]";
            case ERR_IO:
                return base + " [ERR_IO: file I/O or descriptor failure]";
            case ERR_CORRUPT_GGUF:
                return base + " [ERR_CORRUPT_GGUF: corrupt or unsupported GGUF format]";
            case ERR_OOM:
                return base + " [ERR_OOM: host system out of memory]";
            case ERR_CTX_FULL:
                return base + " [ERR_CTX_FULL: sequence context window exhausted]";
            case ERR_CHECKPOINT:
                return base + " [ERR_CHECKPOINT: state checkpoint serialization error]";
            default:
                return base + " [Error code: " + code + "]";
        }
    }
}

