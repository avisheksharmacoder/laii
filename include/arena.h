#ifndef LFM_ARENA_H
#define LFM_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define LFM_CACHE_LINE_SIZE 64

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t offset;
} MemoryArena;

// Initializes an arena with pre-allocated memory aligned to cache lines
bool arena_init(MemoryArena *arena, size_t capacity);

// Allocates memory from the arena aligned to 64 bytes
void* arena_alloc(MemoryArena *arena, size_t size);

// Resets arena offset to 0 without freeing backing storage
void arena_reset(MemoryArena *arena);

// Releases backing storage
void arena_free(MemoryArena *arena);

// Gets current memory usage in bytes
static inline size_t arena_used(const MemoryArena *arena) {
    return arena ? arena->offset : 0;
}

// Gets remaining memory in bytes
static inline size_t arena_remaining(const MemoryArena *arena) {
    return (arena && arena->capacity >= arena->offset) ? (arena->capacity - arena->offset) : 0;
}

#endif // LFM_ARENA_H
