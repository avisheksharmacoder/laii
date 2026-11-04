#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include "arena.h"

int main(void) {
    printf("[Test] Running Memory Arena allocator tests...\n");

    MemoryArena arena;
    size_t capacity = 1024 * 64; // 64 KB

    // 1. Initialize
    if (!arena_init(&arena, capacity)) {
        fprintf(stderr, "FAILED: arena_init failed\n");
        return 1;
    }
    assert(arena.buffer != NULL);
    assert(arena.offset == 0);
    assert(arena.capacity >= capacity);

    // 2. Alignment Test: Every allocation must be 64-byte aligned
    size_t test_sizes[] = { 1, 3, 17, 32, 63, 64, 65, 128, 255, 1024 };
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(test_sizes[0]); ++i) {
        void *p = arena_alloc(&arena, test_sizes[i]);
        if (!p) {
            fprintf(stderr, "FAILED: arena_alloc returned NULL for size %zu\n", test_sizes[i]);
            arena_free(&arena);
            return 1;
        }
        uintptr_t addr = (uintptr_t)p;
        if (addr % LFM_CACHE_LINE_SIZE != 0) {
            fprintf(stderr, "FAILED: Allocation of size %zu is not 64-byte aligned (addr: %p)!\n",
                    test_sizes[i], p);
            arena_free(&arena);
            return 1;
        }
    }
    printf("  All %zu allocation sizes strictly aligned to 64 bytes.\n",
           sizeof(test_sizes)/sizeof(test_sizes[0]));

    // 3. Reset test
    arena_reset(&arena);
    if (arena.offset != 0) {
        fprintf(stderr, "FAILED: arena_reset did not zero the offset (offset=%zu)\n", arena.offset);
        arena_free(&arena);
        return 1;
    }
    printf("  Arena reset verified.\n");

    // 4. Boundary / OOM Handling test
    void *large_alloc = arena_alloc(&arena, arena.capacity + 64);
    if (large_alloc != NULL) {
        fprintf(stderr, "FAILED: arena_alloc should return NULL on OOM!\n");
        arena_free(&arena);
        return 1;
    }
    printf("  Out-of-memory boundary correctly handled (returned NULL).\n");

    // 5. Cleanup
    arena_free(&arena);
    assert(arena.buffer == NULL);
    assert(arena.capacity == 0);
    assert(arena.offset == 0);

    printf("PASSED: Memory Arena tests successful!\n");
    return 0;
}
