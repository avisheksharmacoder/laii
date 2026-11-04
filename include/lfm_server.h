#ifndef LFM_SERVER_H
#define LFM_SERVER_H

#include "lfm_c_api.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    const char *host;
    int port;
} LFMServerConfig;

// Starts the single-binary HTTP/REST server and embedded Web UI.
// Listens on config->port until SIGINT or SIGTERM is received.
int lfm_server_start(lfm_engine_t *engine, const LFMServerConfig *config);

#endif // LFM_SERVER_H
