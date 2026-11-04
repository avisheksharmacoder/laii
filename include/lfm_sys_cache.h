#ifndef LFM_SYS_CACHE_H
#define LFM_SYS_CACHE_H

#ifndef LFM2_SYS_CACHE_NO_MAIN
#define LFM2_SYS_CACHE_NO_MAIN 1
#endif

#if __has_include("tools/lfm2_sys_cache.c")
#include "tools/lfm2_sys_cache.c"
#elif __has_include("lfm2_sys_cache.c")
#include "lfm2_sys_cache.c"
#else
#include "../tools/lfm2_sys_cache.c"
#endif

#endif // LFM_SYS_CACHE_H

