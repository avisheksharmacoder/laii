#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return sizeof(float);
        case GGML_TYPE_F16:  return sizeof(uint16_t);
        case GGML_TYPE_BF16: return sizeof(uint16_t);
        case GGML_TYPE_I8:   return sizeof(int8_t);
        case GGML_TYPE_I16:  return sizeof(int16_t);
        case GGML_TYPE_I32:  return sizeof(int32_t);
        case GGML_TYPE_I64:  return sizeof(int64_t);
        case GGML_TYPE_F64:  return sizeof(double);
        // Quantized block formats (bytes per block / block size)
        case GGML_TYPE_Q4_0: return 18; // 32 values per 18 bytes
        case GGML_TYPE_Q4_1: return 20; // 32 values per 20 bytes
        case GGML_TYPE_Q5_0: return 22; // 32 values per 22 bytes
        case GGML_TYPE_Q5_1: return 24; // 32 values per 24 bytes
        case GGML_TYPE_Q8_0: return 34; // 32 values per 34 bytes (2-byte f16 scale + 32 int8)
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q4_K: return 144; // 256 values per 144 bytes
        case GGML_TYPE_Q5_K: return 176; // 256 values per 176 bytes
        case GGML_TYPE_Q6_K: return 210; // 256 values per 210 bytes
        default:             return 4;
    }
}

static bool read_bytes(const uint8_t *base, size_t *cursor, size_t file_size, void *dst, size_t n) {
    if (*cursor + n > file_size) return false;
    memcpy(dst, base + *cursor, n);
    *cursor += n;
    return true;
}

static char* read_string(const uint8_t *base, size_t *cursor, size_t file_size, uint64_t *out_len) {
    uint64_t len = 0;
    if (!read_bytes(base, cursor, file_size, &len, sizeof(uint64_t))) return NULL;
    if (*cursor + len > file_size) return NULL;

    char *str = (char*)malloc(len + 1);
    if (!str) return NULL;
    memcpy(str, base + *cursor, len);
    str[len] = '\0';
    *cursor += len;
    if (out_len) *out_len = len;
    return str;
}

static bool read_value(const uint8_t *base, size_t *cursor, size_t file_size, GGUFType type, GGUFValue *val) {
    val->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   return read_bytes(base, cursor, file_size, &val->val_u8, 1);
        case GGUF_TYPE_INT8:    return read_bytes(base, cursor, file_size, &val->val_i8, 1);
        case GGUF_TYPE_UINT16:  return read_bytes(base, cursor, file_size, &val->val_u16, 2);
        case GGUF_TYPE_INT16:   return read_bytes(base, cursor, file_size, &val->val_i16, 2);
        case GGUF_TYPE_UINT32:  return read_bytes(base, cursor, file_size, &val->val_u32, 4);
        case GGUF_TYPE_INT32:   return read_bytes(base, cursor, file_size, &val->val_i32, 4);
        case GGUF_TYPE_FLOAT32: return read_bytes(base, cursor, file_size, &val->val_f32, 4);
        case GGUF_TYPE_BOOL:    return read_bytes(base, cursor, file_size, &val->val_bool, 1);
        case GGUF_TYPE_UINT64:  return read_bytes(base, cursor, file_size, &val->val_u64, 8);
        case GGUF_TYPE_INT64:   return read_bytes(base, cursor, file_size, &val->val_i64, 8);
        case GGUF_TYPE_FLOAT64: return read_bytes(base, cursor, file_size, &val->val_f64, 8);
        case GGUF_TYPE_STRING: {
            uint64_t len = 0;
            char *str = read_string(base, cursor, file_size, &len);
            if (!str) return false;
            val->val_str.data = str;
            val->val_str.len = len;
            return true;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t item_type = 0;
            uint64_t item_count = 0;
            if (!read_bytes(base, cursor, file_size, &item_type, 4)) return false;
            if (!read_bytes(base, cursor, file_size, &item_count, 8)) return false;

            val->val_arr.type = (GGUFType)item_type;
            val->val_arr.len = item_count;

            if (item_type == GGUF_TYPE_STRING) {
                GGUFString *strs = (GGUFString*)calloc(item_count, sizeof(GGUFString));
                for (uint64_t i = 0; i < item_count; ++i) {
                    uint64_t slen = 0;
                    strs[i].data = read_string(base, cursor, file_size, &slen);
                    strs[i].len = slen;
                }
                val->val_arr.data = strs;
            } else {
                size_t elem_sz = 1;
                switch (item_type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_BOOL:    elem_sz = 1; break;
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:   elem_sz = 2; break;
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32: elem_sz = 4; break;
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: elem_sz = 8; break;
                    default: elem_sz = 4; break;
                }
                size_t total_sz = item_count * elem_sz;
                void *arr_buf = malloc(total_sz > 0 ? total_sz : 1);
                if (!read_bytes(base, cursor, file_size, arr_buf, total_sz)) {
                    free(arr_buf);
                    return false;
                }
                val->val_arr.data = arr_buf;
            }
            return true;
        }
        default:
            fprintf(stderr, "[GGUF] Unknown GGUFType: %d\n", type);
            return false;
    }
}

static void free_value(GGUFValue *val) {
    if (!val) return;
    if (val->type == GGUF_TYPE_STRING) {
        free(val->val_str.data);
    } else if (val->type == GGUF_TYPE_ARRAY) {
        if (val->val_arr.type == GGUF_TYPE_STRING && val->val_arr.data) {
            GGUFString *strs = (GGUFString*)val->val_arr.data;
            for (uint64_t i = 0; i < val->val_arr.len; ++i) {
                free(strs[i].data);
            }
        }
        free(val->val_arr.data);
    }
}

bool gguf_open(GGUFFile *file, const char *filepath) {
    if (!file || !filepath) return false;
    memset(file, 0, sizeof(GGUFFile));
    file->alignment = 32;

    file->fd = open(filepath, O_RDONLY);
    if (file->fd < 0) {
        fprintf(stderr, "[GGUF] Failed to open file: %s\n", filepath);
        return false;
    }

    struct stat sb;
    if (fstat(file->fd, &sb) < 0) {
        close(file->fd);
        return false;
    }
    file->file_size = (size_t)sb.st_size;

    file->mmap_addr = mmap(NULL, file->file_size, PROT_READ, MAP_SHARED, file->fd, 0);
    if (file->mmap_addr == MAP_FAILED) {
        fprintf(stderr, "[GGUF] mmap failed for file %s\n", filepath);
        close(file->fd);
        return false;
    }

    // Advise sequential access to pre-populate kernel page caches
    madvise(file->mmap_addr, file->file_size, MADV_SEQUENTIAL);

    const uint8_t *base = (const uint8_t*)file->mmap_addr;
    size_t cursor = 0;

    // 1. Read Header
    uint32_t magic = 0;
    if (!read_bytes(base, &cursor, file->file_size, &magic, 4) || magic != GGUF_MAGIC) {
        fprintf(stderr, "[GGUF] Invalid magic: 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        gguf_close(file);
        return false;
    }

    if (!read_bytes(base, &cursor, file->file_size, &file->version, 4)) {
        gguf_close(file);
        return false;
    }

    if (file->version != 2 && file->version != 3) {
        fprintf(stderr, "[GGUF] Unsupported GGUF version: %u\n", file->version);
        gguf_close(file);
        return false;
    }

    if (!read_bytes(base, &cursor, file->file_size, &file->n_tensors, 8) ||
        !read_bytes(base, &cursor, file->file_size, &file->n_kv, 8)) {
        gguf_close(file);
        return false;
    }

    // 2. Read Metadata Key-Values
    file->kv = (GGUFKV*)calloc(file->n_kv, sizeof(GGUFKV));
    for (uint64_t i = 0; i < file->n_kv; ++i) {
        uint64_t klen = 0;
        file->kv[i].key = read_string(base, &cursor, file->file_size, &klen);
        if (!file->kv[i].key) {
            gguf_close(file);
            return false;
        }

        uint32_t vtype = 0;
        if (!read_bytes(base, &cursor, file->file_size, &vtype, 4)) {
            gguf_close(file);
            return false;
        }

        if (!read_value(base, &cursor, file->file_size, (GGUFType)vtype, &file->kv[i].val)) {
            gguf_close(file);
            return false;
        }
    }

    // Check for custom alignment
    const GGUFValue *align_val = gguf_get_val(file, "general.alignment");
    if (align_val && align_val->type == GGUF_TYPE_UINT32) {
        file->alignment = align_val->val_u32;
    }

    // 3. Read Tensor Info Directory
    file->tensors = (GGUFTensor*)calloc(file->n_tensors, sizeof(GGUFTensor));
    for (uint64_t i = 0; i < file->n_tensors; ++i) {
        uint64_t nlen = 0;
        file->tensors[i].name = read_string(base, &cursor, file->file_size, &nlen);
        if (!file->tensors[i].name) {
            gguf_close(file);
            return false;
        }

        if (!read_bytes(base, &cursor, file->file_size, &file->tensors[i].n_dims, 4)) {
            gguf_close(file);
            return false;
        }

        for (uint32_t d = 0; d < file->tensors[i].n_dims && d < 4; ++d) {
            if (!read_bytes(base, &cursor, file->file_size, &file->tensors[i].dims[d], 8)) {
                gguf_close(file);
                return false;
            }
        }

        uint32_t t_type = 0;
        if (!read_bytes(base, &cursor, file->file_size, &t_type, 4) ||
            !read_bytes(base, &cursor, file->file_size, &file->tensors[i].offset, 8)) {
            gguf_close(file);
            return false;
        }
        file->tensors[i].type = (GGMLType)t_type;
    }

    // 4. Align cursor to data segment
    size_t align = file->alignment > 0 ? file->alignment : 32;
    size_t data_offset = (cursor + align - 1) & ~(align - 1);
    if (data_offset > file->file_size) {
        fprintf(stderr, "[GGUF] Corrupted data offset: %zu > %zu\n", data_offset, file->file_size);
        gguf_close(file);
        return false;
    }

    file->data_segment = (uint8_t*)file->mmap_addr + data_offset;
    file->data_size = file->file_size - data_offset;

    // Attach data pointers and compute tensor sizes
    for (uint64_t i = 0; i < file->n_tensors; ++i) {
        file->tensors[i].data = file->data_segment + file->tensors[i].offset;

        // Calculate tensor size
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < file->tensors[i].n_dims; ++d) {
            n_elements *= file->tensors[i].dims[d];
        }

        if (file->tensors[i].type == GGML_TYPE_F32) {
            file->tensors[i].size_bytes = n_elements * sizeof(float);
        } else if (file->tensors[i].type == GGML_TYPE_F16) {
            file->tensors[i].size_bytes = n_elements * sizeof(uint16_t);
        } else if (file->tensors[i].type == GGML_TYPE_Q8_0) {
            file->tensors[i].size_bytes = (n_elements / 32) * 34;
        } else {
            file->tensors[i].size_bytes = n_elements * 2; // conservative approximation
        }
    }

    return true;
}

void gguf_close(GGUFFile *file) {
    if (!file) return;

    if (file->kv) {
        for (uint64_t i = 0; i < file->n_kv; ++i) {
            free(file->kv[i].key);
            free_value(&file->kv[i].val);
        }
        free(file->kv);
        file->kv = NULL;
    }

    if (file->tensors) {
        for (uint64_t i = 0; i < file->n_tensors; ++i) {
            free(file->tensors[i].name);
        }
        free(file->tensors);
        file->tensors = NULL;
    }

    if (file->mmap_addr && file->mmap_addr != MAP_FAILED) {
        munmap(file->mmap_addr, file->file_size);
        file->mmap_addr = NULL;
    }

    if (file->fd >= 0) {
        close(file->fd);
        file->fd = -1;
    }
}

const GGUFValue* gguf_get_val(const GGUFFile *file, const char *key) {
    if (!file || !file->kv || !key) return NULL;
    for (uint64_t i = 0; i < file->n_kv; ++i) {
        if (file->kv[i].key && strcmp(file->kv[i].key, key) == 0) {
            return &file->kv[i].val;
        }
    }
    return NULL;
}

uint32_t gguf_get_u32(const GGUFFile *file, const char *key, uint32_t default_val) {
    const GGUFValue *val = gguf_get_val(file, key);
    if (!val) return default_val;
    if (val->type == GGUF_TYPE_UINT32) return val->val_u32;
    if (val->type == GGUF_TYPE_INT32) return (uint32_t)val->val_i32;
    if (val->type == GGUF_TYPE_UINT64) return (uint32_t)val->val_u64;
    return default_val;
}

uint64_t gguf_get_u64(const GGUFFile *file, const char *key, uint64_t default_val) {
    const GGUFValue *val = gguf_get_val(file, key);
    if (!val) return default_val;
    if (val->type == GGUF_TYPE_UINT64) return val->val_u64;
    if (val->type == GGUF_TYPE_UINT32) return (uint64_t)val->val_u32;
    if (val->type == GGUF_TYPE_INT64) return (uint64_t)val->val_i64;
    return default_val;
}

float gguf_get_f32(const GGUFFile *file, const char *key, float default_val) {
    const GGUFValue *val = gguf_get_val(file, key);
    if (!val) return default_val;
    if (val->type == GGUF_TYPE_FLOAT32) return val->val_f32;
    if (val->type == GGUF_TYPE_FLOAT64) return (float)val->val_f64;
    return default_val;
}

const char* gguf_get_str(const GGUFFile *file, const char *key, const char *default_val) {
    const GGUFValue *val = gguf_get_val(file, key);
    if (!val || val->type != GGUF_TYPE_STRING) return default_val;
    return val->val_str.data;
}

const GGUFTensor* gguf_get_tensor(const GGUFFile *file, const char *name) {
    if (!file || !file->tensors || !name) return NULL;
    for (uint64_t i = 0; i < file->n_tensors; ++i) {
        if (file->tensors[i].name && strcmp(file->tensors[i].name, name) == 0) {
            return &file->tensors[i];
        }
    }
    return NULL;
}
