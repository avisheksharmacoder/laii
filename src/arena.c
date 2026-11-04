#define _POSIX_C_SOURCE 200809L
#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

bool arena_init(MemoryArena *arena, size_t capacity) {
    if (!arena || capacity == 0) return false;

    // Align total capacity up to cache line multiple
    size_t aligned_cap = (capacity + LFM_CACHE_LINE_SIZE - 1) & ~(LFM_CACHE_LINE_SIZE - 1);

    void *ptr = NULL;
    int ret = posix_memalign(&ptr, LFM_CACHE_LINE_SIZE, aligned_cap);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "[Arena] Failed to allocate aligned memory of size %zu\n", aligned_cap);
        arena->buffer = NULL;
        arena->capacity = 0;
        arena->offset = 0;
        return false;
    }

    memset(ptr, 0, aligned_cap);
    arena->buffer = (uint8_t*)ptr;
    arena->capacity = aligned_cap;
    arena->offset = 0;
    return true;
}

void* arena_alloc(MemoryArena *arena, size_t size) {
    if (!arena || !arena->buffer || size == 0) return NULL;

    // Align size up to 64-byte boundary
    size_t aligned_size = (size + LFM_CACHE_LINE_SIZE - 1) & ~(LFM_CACHE_LINE_SIZE - 1);

    if (arena->offset + aligned_size > arena->capacity) {
        fprintf(stderr, "[Arena] Out of memory! Requested %zu (aligned %zu), used %zu / %zu\n",
                size, aligned_size, arena->offset, arena->capacity);
        return NULL;
    }

    void *ptr = arena->buffer + arena->offset;
    arena->offset += aligned_size;
    return ptr;
}

void arena_reset(MemoryArena *arena) {
    if (arena) {
        arena->offset = 0;
    }
}

void arena_free(MemoryArena *arena) {
    if (arena && arena->buffer) {
        free(arena->buffer);
        arena->buffer = NULL;
        arena->capacity = 0;
        arena->offset = 0;
    }
}
