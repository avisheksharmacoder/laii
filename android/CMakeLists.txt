# CMakeLists.txt - Android Studio NDK Build for LFM2 Native Engine
#
# Builds 16 KB page-aligned `liblfm2.so` with ARM NEON + DotProd SIMD kernels.
# Target: Android API 26+ (arm64-v8a)

cmake_minimum_required(VERSION 3.22.1)
project("lfm2" C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source directory paths relative to CMakeLists.txt
set(LFM2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories(
    "${LFM2_ROOT}/include"
    "${LFM2_ROOT}/src"
)

# Core engine source files
add_library(lfm2 SHARED
    "${LFM2_ROOT}/src/lfm2_api.c"
)

# Compiler optimization flags
target_compile_definitions(lfm2 PRIVATE
    _GNU_SOURCE
    LFM2_ENABLE_JNI=1
)

target_compile_options(lfm2 PRIVATE
    -O3
    -fPIC
    -fvisibility=hidden
    -Wall
    -Wextra
    -Wno-unused-function
)

# Android ARM64 hardware features
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(lfm2 PRIVATE
        -march=armv8.2-a+dotprod+fp16
    )
endif()

# Android 15+ (API 35) 16 KB ELF segment alignment
target_link_options(lfm2 PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,--exclude-libs,ALL
)

# Android system libraries (libm for math, liblog for logcat)
target_link_libraries(lfm2 PRIVATE
    m
    log
)

