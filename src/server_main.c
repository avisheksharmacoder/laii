#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "lfm_c_api.h"
#include "lfm_server.h"
#include "lfm2_simd_avx2.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -m <model.gguf> [options]\n\n"
        "Liquid AI (LFM2) Single-Binary HTTP Server & Web Studio\n\n"
        "Options:\n"
        "  -m, --model <path>     Path to LFM2 GGUF model file (required)\n"
        "  -p, --port <int>       HTTP port to listen on (default: 8080)\n"
        "  -a, --host <ip>        Host/IP to bind to (default: 0.0.0.0)\n"
        "  -j, --threads <int>    Number of CPU worker threads (default: auto)\n"
        "  -h, --help             Show this help menu\n\n"
        "Example:\n"
        "  %s -m models/lfm2-1.2b.gguf -p 8080 -j 4\n",
        prog, prog);
}

int main(int argc, char **argv) {
    lfm2_set_denormals_zero_mode();
    const char *model_path = NULL;
    const char *host = "0.0.0.0";
    int port = 8080;
    int num_threads = 0;

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--host") == 0) && i + 1 < argc) {
            host = argv[++i];
        } else if ((strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: Model path (-m, --model) is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Initializing Liquid AI model from %s ...\n", model_path);
    lfm_engine_t *engine = lfm_create(model_path, num_threads);
    if (!engine) {
        fprintf(stderr, "Error: Failed to initialize engine with model %s\n", model_path);
        return 1;
    }

    LFMServerConfig cfg = {
        .host = host,
        .port = port
    };

    int ret = lfm_server_start(engine, &cfg);
    lfm_destroy(engine);
    return ret;
}
