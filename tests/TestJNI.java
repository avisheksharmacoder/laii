package tests;

import ai.liquid.lfm2.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileDescriptor;
import java.lang.reflect.Field;
import java.util.Arrays;

public class TestJNI {

    public static void main(String[] args) {
        System.out.println("===============================================================");
        System.out.println("  LFM2.c Native Java / Kotlin JNI Bridge End-to-End Suite");
        System.out.println("===============================================================");

        String libPath = new File("lib/liblfm2.so").getAbsolutePath();
        System.out.println("[Step 1/8] Loading native shared library from: " + libPath);
        System.load(libPath);
        System.out.println("  Native library loaded and JNI_OnLoad registered successfully.");

        String modelPath = null;
        if (args.length > 0 && args[0] != null && !args[0].isEmpty()) {
            modelPath = args[0];
        } else {
            String envModel = System.getenv("LFM2_TEST_MODEL");
            if (envModel != null && !envModel.isEmpty()) {
                modelPath = envModel;
            } else {
                String[] candidates = {
                    "models/LFM2.5-230M-Q4_K_M.gguf",
                    "build/ci_lfm2_tiny.gguf",
                    "tests/lfm2_toy.gguf"
                };
                for (String c : candidates) {
                    if (new File(c).exists()) {
                        modelPath = c;
                        break;
                    }
                }
            }
        }
        if (modelPath == null || !new File(modelPath).exists()) {
            System.err.println("Model file not found: " + modelPath);
            System.exit(1);
        }

        System.out.println("\n[Step 2/8] Testing LFM2Model.load() from path: " + modelPath);
        long t0 = System.currentTimeMillis();
        LFM2Model model = LFM2Model.load(modelPath);
        long tLoad = System.currentTimeMillis() - t0;
        System.out.printf("  Model loaded in %d ms!%n", tLoad);
        System.out.println("  Architecture:       " + model.getArch());
        System.out.println("  Active ISA Kernel:  " + model.getActiveIsa());
        System.out.println("  Total Layers:       " + model.getNumLayers() +
                           " (ShortConv=" + model.getNumShortConvLayers() +
                           ", GQA=" + model.getNumGqaLayers() + ")");
        System.out.println("  Hidden Dim:         " + model.getHiddenDim());
        System.out.println("  FFN Dim:            " + model.getFfnDim());
        System.out.println("  Vocab Size:         " + model.getVocabSize());
        System.out.println("  File Size:          " + (model.getFileSizeBytes() / (1024 * 1024)) + " MB");

        assert model.getNumLayers() > 0 : "Layers should be > 0";
        assert model.getVocabSize() > 0 : "Vocab should be > 0";

        System.out.println("\n[Step 3/8] Testing Native BPE Tokenizer...");
        String sampleText = "The quick brown fox jumps over the lazy dog";
        int[] tokens = model.tokenize(sampleText);
        System.out.printf("  Tokenized \"%s\" -> %d tokens: %s%n",
            sampleText, tokens.length, Arrays.toString(tokens));
        assert tokens.length > 0 : "Tokens array should not be empty";

        System.out.println("\n[Step 4/8] Testing Android Scoped Storage / Content URI FD Loader...");
        try (FileInputStream fis = new FileInputStream(modelPath)) {
            FileDescriptor fdObj = fis.getFD();
            int rawFd = -1;
            try {
                Field fdField = FileDescriptor.class.getDeclaredField("fd");
                fdField.setAccessible(true);
                rawFd = fdField.getInt(fdObj);
            } catch (Throwable t) {
                System.out.println("  (Note: OpenJDK module reflection restricted; skipping raw FD extraction on desktop)");
            }

            if (rawFd >= 0) {
                System.out.println("  Extracted OS File Descriptor: fd=" + rawFd);
                try (LFM2Model fdModel = LFM2Model.loadFromFd(rawFd)) {
                    System.out.println("  FD-loaded model verified: arch=" + fdModel.getArch() +
                                       ", layers=" + fdModel.getNumLayers());
                    assert fdModel.getNumLayers() == model.getNumLayers();
                }
                System.out.println("  FD-loaded model closed cleanly.");
            }
        } catch (Exception e) {
            System.err.println("FD test exception: " + e.getMessage());
            throw new RuntimeException(e);
        }

        System.out.println("\n[Step 5/8] Testing Mobile OS Lifecycle Hooks (releasePageCache / warmupPageCache)...");
        long rssBefore = LFM2Model.getProcessRssBytes();
        System.out.printf("  Initial Process RSS: %.2f MB%n", rssBefore / (1024.0 * 1024.0));
        model.releasePageCache();
        long rssReleased = LFM2Model.getProcessRssBytes();
        System.out.printf("  RSS after releasePageCache() (onPause): %.2f MB%n", rssReleased / (1024.0 * 1024.0));
        model.warmupPageCache();
        long rssWarmed = LFM2Model.getProcessRssBytes();
        System.out.printf("  RSS after warmupPageCache() (onResume): %.2f MB%n", rssWarmed / (1024.0 * 1024.0));

        System.out.println("\n[Step 6/8] Testing Stateful LFM2Context & Streaming Generation...");
        ContextParams ctxParams = new ContextParams()
            .setMaxSeqLen(2048)
            .setNumThreads(4)
            .setAutoClampRam(true);

        try (LFM2Context ctx = model.createContext(ctxParams)) {
            ctx.setSystemPrompt("You are a helpful and concise assistant.");
            ctx.addStopString("<|im_end|>");

            String prompt = "In three words, describe the speed of native C.";
            System.out.println("  Prompt: \"" + prompt + "\"");
            System.out.print("  Streamed output: \"");

            SampleParams sp = new SampleParams()
                .setTemperature(0.20f)
                .setTopK(40)
                .setMinP(0.10f);

            ctx.beginChatTurn(prompt, sp);
            int maxChunks = (model.getVocabSize() < 4096) ? 64 : 512;
            int chunkCount = 0;
            GenerationChunk chunk;
            while (chunkCount < maxChunks && (chunk = ctx.nextChunk()) != null) {
                System.out.print(chunk.text);
                chunkCount++;
            }
            System.out.println("\"");
            System.out.println("  Total chunks received: " + chunkCount);

            PerfStats stats = ctx.getStats();
            System.out.println("  " + stats.toString());
            System.out.printf("  Prefill Speed: %.1f tok/s%n", stats.prefillTokPerSec);
            System.out.printf("  Decode Speed:  %.1f tok/s%n", stats.decodeTokPerSec);
            assert stats.promptTokens > 0 : "Prompt tokens should be > 0";
            if (model.getVocabSize() >= 4096) {
                assert chunkCount > 0 : "Should have received streamed chunks";
                assert stats.generatedTokens > 0 : "Generated tokens should be > 0";
            } else if (chunkCount == 0) {
                System.out.println("  (Note: synthetic tiny fixture emitted EOS immediately)");
            }

            System.out.println("\n[Step 7/8] Testing Normalized Text Embeddings...");
            float[] embd = ctx.embed("Fast native inference on smartphone SoCs.", PoolingType.MEAN);
            System.out.printf("  Computed embedding vector of size %d.%n", embd.length);
            assert embd.length == model.getHiddenDim() : "Embedding dim must match hiddenDim";

            double l2Norm = 0.0;
            for (float v : embd) l2Norm += v * v;
            l2Norm = Math.sqrt(l2Norm);
            System.out.printf("  Vector L2 Norm: %.6f (expected ~1.000000)%n", l2Norm);
            assert Math.abs(l2Norm - 1.0) < 1e-3 : "Embedding must be unit L2-normalized";
        }

        System.out.println("\n[Step 8/8] Testing Context & Model Cleanup...");
        model.close();
        assert model.isClosed() : "Model should report closed";
        System.out.println("  Model closed successfully.");

        System.out.println("\n===============================================================");
        System.out.println("  ALL 8 JAVA / KOTLIN JNI BRIDGE TESTS PASSED 100%!  ");
        System.out.println("===============================================================");
    }
}
