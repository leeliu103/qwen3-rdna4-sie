#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GGUF_MAX_DIMS 8

typedef struct {
    const char *ptr;
    uint64_t len;
} gguf_str;

typedef struct {
    gguf_str key;
    uint32_t type;
    uint64_t value_pos;
} gguf_kv;

typedef struct {
    gguf_str name;
    uint32_t ndim;
    uint64_t dim[GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} gguf_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;

    gguf_kv *kv;
    gguf_tensor *tensors;
} gguf_file;

bool gguf_open(gguf_file *g, const char *path, char *err, size_t err_size);
void gguf_close(gguf_file *g);

bool gguf_get_u32(const gguf_file *g, const char *key, uint32_t *out);
bool gguf_get_u64(const gguf_file *g, const char *key, uint64_t *out);
bool gguf_get_f32(const gguf_file *g, const char *key, float *out);
bool gguf_get_string(const gguf_file *g, const char *key, gguf_str *out);

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name);
const void *gguf_tensor_data(const gguf_file *g, const gguf_tensor *t);

const char *gguf_type_name(uint32_t type);

#endif
