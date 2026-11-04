#ifndef LFM_GGUF_H
#define LFM_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define GGUF_MAGIC 0x46554747 // "GGUF" in little-endian

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_BF16 = 16,
    GGML_TYPE_I8   = 24,
    GGML_TYPE_I16  = 25,
    GGML_TYPE_I32  = 26,
    GGML_TYPE_I64  = 27,
    GGML_TYPE_F64  = 28,
} GGMLType;

typedef struct {
    char *data;
    uint64_t len;
} GGUFString;

typedef struct GGUFArray {
    GGUFType type;
    uint64_t len;
    void *data;
} GGUFArray;

typedef struct {
    GGUFType type;
    union {
        uint8_t   val_u8;
        int8_t    val_i8;
        uint16_t  val_u16;
        int16_t   val_i16;
        uint32_t  val_u32;
        int32_t   val_i32;
        float     val_f32;
        bool      val_bool;
        GGUFString val_str;
        GGUFArray  val_arr;
        uint64_t  val_u64;
        int64_t   val_i64;
        double    val_f64;
    };
} GGUFValue;

typedef struct {
    char *key;
    GGUFValue val;
} GGUFKV;

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    GGMLType type;
    uint64_t offset;
    void *data; // Direct pointer into mmap data segment
    size_t size_bytes;
} GGUFTensor;

typedef struct {
    int fd;
    void *mmap_addr;
    size_t file_size;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint32_t alignment;

    GGUFKV *kv;
    GGUFTensor *tensors;

    uint8_t *data_segment;
    size_t data_size;
} GGUFFile;

// Load GGUF file via mmap
bool gguf_open(GGUFFile *file, const char *filepath);

// Free mmap and parsed structures
void gguf_close(GGUFFile *file);

// Find metadata KV by key
const GGUFValue* gguf_get_val(const GGUFFile *file, const char *key);
uint32_t gguf_get_u32(const GGUFFile *file, const char *key, uint32_t default_val);
uint64_t gguf_get_u64(const GGUFFile *file, const char *key, uint64_t default_val);
float gguf_get_f32(const GGUFFile *file, const char *key, float default_val);
const char* gguf_get_str(const GGUFFile *file, const char *key, const char *default_val);

// Find tensor by exact name
const GGUFTensor* gguf_get_tensor(const GGUFFile *file, const char *name);

// Returns element size or block size in bytes for a given GGMLType
size_t ggml_type_size(GGMLType type);

#endif // LFM_GGUF_H
