#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "gguf.h"

int main(void) {
    printf("[Test] Running GGUF binary parser and mmap loader test...\n");

    GGUFFile file;
    if (!gguf_open(&file, "tests/lfm2_toy.gguf")) {
        fprintf(stderr, "FAILED: Could not open tests/lfm2_toy.gguf\n");
        return 1;
    }

    printf("  GGUF Version: %u\n", file.version);
    printf("  Tensors:      %lu\n", (unsigned long)file.n_tensors);
    printf("  Metadata KV:  %lu\n", (unsigned long)file.n_kv);

    if (file.version != 3) {
        fprintf(stderr, "FAILED: Unexpected version %u\n", file.version);
        gguf_close(&file);
        return 1;
    }

    if (file.n_tensors != 22) {
        fprintf(stderr, "FAILED: Expected 22 tensors, got %lu\n", (unsigned long)file.n_tensors);
        gguf_close(&file);
        return 1;
    }

    const char *arch = gguf_get_str(&file, "general.architecture", "");
    printf("  Architecture: %s\n", arch);
    if (strcmp(arch, "lfm2") != 0) {
        fprintf(stderr, "FAILED: Expected architecture 'lfm2', got '%s'\n", arch);
        gguf_close(&file);
        return 1;
    }

    uint32_t hidden_dim = gguf_get_u32(&file, "lfm2.embedding_length", 0);
    uint32_t n_layers   = gguf_get_u32(&file, "lfm2.block_count", 0);
    printf("  Hidden Dim:   %u\n", hidden_dim);
    printf("  Layer Count:  %u\n", n_layers);

    if (hidden_dim != 64 || n_layers != 2) {
        fprintf(stderr, "FAILED: Hyperparameter mismatch!\n");
        gguf_close(&file);
        return 1;
    }

    // Check specific tensor
    const GGUFTensor *conv_t = gguf_get_tensor(&file, "blk.0.shortconv.conv.weight");
    if (!conv_t) {
        fprintf(stderr, "FAILED: Missing blk.0.shortconv.conv.weight!\n");
        gguf_close(&file);
        return 1;
    }
    printf("  Found tensor: %s (offset=%lu, data=%p)\n",
           conv_t->name, (unsigned long)conv_t->offset, conv_t->data);

    gguf_close(&file);
    printf("PASSED: GGUF parser verified!\n");
    return 0;
}
