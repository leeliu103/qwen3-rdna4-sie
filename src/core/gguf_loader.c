#include "gguf_loader.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} gguf_cursor;

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_tensor_types[] = {
    [0]  = {"f32",       1,   4},
    [1]  = {"f16",       1,   2},
    [2]  = {"q4_0",     32,  18},
    [3]  = {"q4_1",     32,  20},
    [6]  = {"q5_0",     32,  22},
    [7]  = {"q5_1",     32,  24},
    [8]  = {"q8_0",     32,  34},
    [9]  = {"q8_1",     32,  36},
    [10] = {"q2_K",    256,  84},
    [11] = {"q3_K",    256, 110},
    [12] = {"q4_K",    256, 144},
    [13] = {"q5_K",    256, 176},
    [14] = {"q6_K",    256, 210},
    [15] = {"q8_K",    256, 292},
    [16] = {"iq2_xxs", 256,  66},
    [17] = {"iq2_xs",  256,  74},
    [18] = {"iq3_xxs", 256,  98},
    [19] = {"iq1_s",   256,  50},
    [20] = {"iq4_nl",   32,  18},
    [21] = {"iq3_s",   256, 110},
    [22] = {"iq2_s",   256,  82},
    [23] = {"iq4_xs",  256, 136},
    [24] = {"i8",        1,   1},
    [25] = {"i16",       1,   2},
    [26] = {"i32",       1,   4},
    [27] = {"i64",       1,   8},
    [28] = {"f64",       1,   8},
    [29] = {"iq1_m",   256,  56},
    [30] = {"bf16",      1,   2},
    [34] = {"tq1_0",   256,  54},
    [35] = {"tq2_0",   256,  66},
    [39] = {"mxfp4",    32,  17},
    [40] = {"nvfp4",    64,  36},
    [41] = {"q1_0",    128,  18},
};

static void set_err(char *err, size_t err_size, const char *fmt, ...) {
    if (!err || err_size == 0) return;

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);
    err[err_size - 1] = '\0';
}

static void cursor_error(gguf_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(gguf_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(gguf_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(gguf_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(gguf_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(gguf_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_f32(gguf_cursor *c, float *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_f64(gguf_cursor *c, double *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(gguf_cursor *c, gguf_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static bool align_up_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (alignment == 0) return false;

    const uint64_t rem = value % alignment;
    if (rem == 0) {
        *out = value;
        return true;
    }

    const uint64_t add = alignment - rem;
    if (value > UINT64_MAX - add) return false;
    *out = value + add;
    return true;
}

static bool gguf_streq(gguf_str s, const char *z) {
    const size_t n = strlen(z);
    return s.len == (uint64_t)n && memcmp(s.ptr, z, n) == 0;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(gguf_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    const uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        gguf_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        const uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static gguf_cursor cursor_at(const gguf_file *g, uint64_t pos) {
    gguf_cursor c;
    c.base = g->map;
    c.size = g->size;
    c.pos = pos;
    c.error[0] = '\0';
    return c;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    const uint32_t n = (uint32_t)(sizeof(gguf_tensor_types) / sizeof(gguf_tensor_types[0]));
    if (type >= n || gguf_tensor_types[type].name == NULL) return NULL;
    return &gguf_tensor_types[type];
}

static bool tensor_nbytes(const gguf_tensor *t, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || info->block_elems == 0 || info->block_bytes == 0) return false;

    uint64_t blocks;
    if (info->block_elems == 1) {
        blocks = t->elements;
    } else {
        if (t->dim[0] % info->block_elems != 0) return false;
        blocks = t->elements / info->block_elems;
    }

    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static gguf_kv *find_kv(const gguf_file *g, const char *key) {
    if (!g || !key) return NULL;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (gguf_streq(g->kv[i].key, key)) return &g->kv[i];
    }
    return NULL;
}

static bool parse_metadata(gguf_file *g, gguf_cursor *c, char *err, size_t err_size) {
    if (g->n_kv > (uint64_t)SIZE_MAX / sizeof(g->kv[0])) {
        set_err(err, err_size, "metadata table is too large");
        return false;
    }

    if (g->n_kv != 0) {
        g->kv = (gguf_kv *)calloc((size_t)g->n_kv, sizeof(g->kv[0]));
        if (!g->kv) {
            set_err(err, err_size, "out of memory allocating metadata table");
            return false;
        }
    }

    g->alignment = 32;

    for (uint64_t i = 0; i < g->n_kv; i++) {
        gguf_kv *kv = &g->kv[i];
        if (!cursor_string(c, &kv->key)) goto cursor_fail;
        if (!cursor_u32(c, &kv->type)) goto cursor_fail;

        kv->value_pos = c->pos;

        if (gguf_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            gguf_cursor tmp = cursor_at(g, kv->value_pos);
            uint32_t alignment = 0;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                g->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) goto cursor_fail;
    }

    return true;

cursor_fail:
    set_err(err, err_size, "%s", c->error[0] ? c->error : "failed to parse metadata");
    return false;
}

static bool parse_tensors(gguf_file *g, gguf_cursor *c, char *err, size_t err_size) {
    if (g->n_tensors > (uint64_t)SIZE_MAX / sizeof(g->tensors[0])) {
        set_err(err, err_size, "tensor table is too large");
        return false;
    }

    if (g->n_tensors != 0) {
        g->tensors = (gguf_tensor *)calloc((size_t)g->n_tensors, sizeof(g->tensors[0]));
        if (!g->tensors) {
            set_err(err, err_size, "out of memory allocating tensor table");
            return false;
        }
    }

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (!cursor_string(c, &t->name)) goto cursor_fail;
        if (!cursor_u32(c, &t->ndim)) goto cursor_fail;
        if (t->ndim == 0 || t->ndim > GGUF_MAX_DIMS) {
            set_err(err, err_size, "tensor %" PRIu64 " has unsupported dimension count %u", i, t->ndim);
            return false;
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) goto cursor_fail;
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                set_err(err, err_size, "tensor %" PRIu64 " element count overflows uint64", i);
                return false;
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) goto cursor_fail;
        if (!cursor_u64(c, &t->rel_offset)) goto cursor_fail;

        if (tensor_type(t->type) == NULL) {
            set_err(err, err_size, "tensor %" PRIu64 " has unsupported GGML type %u", i, t->type);
            return false;
        }
        if (!tensor_nbytes(t, &t->bytes)) {
            set_err(err, err_size, "tensor %" PRIu64 " has invalid shape for GGML type %u", i, t->type);
            return false;
        }
    }

    if (!align_up_u64(c->pos, g->alignment, &g->tensor_data_pos)) {
        set_err(err, err_size, "tensor data offset overflows while applying alignment %" PRIu64, g->alignment);
        return false;
    }

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor *t = &g->tensors[i];
        if (t->rel_offset > UINT64_MAX - g->tensor_data_pos) {
            set_err(err, err_size, "tensor %" PRIu64 " offset overflows uint64", i);
            return false;
        }
        t->abs_offset = g->tensor_data_pos + t->rel_offset;
        if (t->abs_offset > g->size || t->bytes > g->size - t->abs_offset) {
            set_err(err, err_size, "tensor %" PRIu64 " data range points outside GGUF file", i);
            return false;
        }
    }

    return true;

cursor_fail:
    set_err(err, err_size, "%s", c->error[0] ? c->error : "failed to parse tensor directory");
    return false;
}

bool gguf_open(gguf_file *g, const char *path, char *err, size_t err_size) {
    if (err && err_size != 0) err[0] = '\0';
    if (!g) {
        set_err(err, err_size, "gguf_open called with null gguf_file");
        return false;
    }
    if (!path) {
        set_err(err, err_size, "gguf_open called with null path");
        return false;
    }

    memset(g, 0, sizeof(*g));
    g->fd = -1;

    g->fd = open(path, O_RDONLY);
    if (g->fd == -1) {
        set_err(err, err_size, "cannot open '%s': %s", path, strerror(errno));
        return false;
    }

    struct stat st;
    if (fstat(g->fd, &st) == -1) {
        set_err(err, err_size, "cannot stat '%s': %s", path, strerror(errno));
        gguf_close(g);
        return false;
    }
    if (st.st_size <= 0) {
        set_err(err, err_size, "'%s' is empty", path);
        gguf_close(g);
        return false;
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        set_err(err, err_size, "'%s' is too large to map on this platform", path);
        gguf_close(g);
        return false;
    }

    g->size = (uint64_t)st.st_size;
    void *map = mmap(NULL, (size_t)g->size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (map == MAP_FAILED) {
        set_err(err, err_size, "cannot mmap '%s': %s", path, strerror(errno));
        g->map = NULL;
        gguf_close(g);
        return false;
    }
    g->map = (const uint8_t *)map;

    gguf_cursor c = cursor_at(g, 0);
    uint32_t magic = 0;
    if (!cursor_u32(&c, &magic)) goto cursor_fail;
    if (magic != GGUF_MAGIC) {
        set_err(err, err_size, "'%s' is not a GGUF file", path);
        gguf_close(g);
        return false;
    }
    if (!cursor_u32(&c, &g->version)) goto cursor_fail;
    if (!cursor_u64(&c, &g->n_tensors)) goto cursor_fail;
    if (!cursor_u64(&c, &g->n_kv)) goto cursor_fail;

    if (g->version != 3) {
        set_err(err, err_size, "unsupported GGUF version %u; only v3 is supported", g->version);
        gguf_close(g);
        return false;
    }

    if (!parse_metadata(g, &c, err, err_size)) {
        gguf_close(g);
        return false;
    }
    if (!parse_tensors(g, &c, err, err_size)) {
        gguf_close(g);
        return false;
    }

    return true;

cursor_fail:
    set_err(err, err_size, "%s", c.error[0] ? c.error : "failed to parse GGUF header");
    gguf_close(g);
    return false;
}

void gguf_close(gguf_file *g) {
    if (!g) return;

    free(g->kv);
    free(g->tensors);
    if (g->map) munmap((void *)g->map, (size_t)g->size);
    if (g->fd >= 0) close(g->fd);

    memset(g, 0, sizeof(*g));
    g->fd = -1;
}

bool gguf_get_u32(const gguf_file *g, const char *key, uint32_t *out) {
    if (!out) return false;
    gguf_kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;

    gguf_cursor c = cursor_at(g, kv->value_pos);
    return cursor_u32(&c, out);
}

bool gguf_get_u64(const gguf_file *g, const char *key, uint64_t *out) {
    if (!out) return false;
    gguf_kv *kv = find_kv(g, key);
    if (!kv) return false;

    gguf_cursor c = cursor_at(g, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

bool gguf_get_f32(const gguf_file *g, const char *key, float *out) {
    if (!out) return false;
    gguf_kv *kv = find_kv(g, key);
    if (!kv) return false;

    gguf_cursor c = cursor_at(g, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) return cursor_f32(&c, out);
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_f64(&c, &v)) return false;
        if (v > (double)FLT_MAX || v < -(double)FLT_MAX) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

bool gguf_get_string(const gguf_file *g, const char *key, gguf_str *out) {
    if (!out) return false;
    gguf_kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;

    gguf_cursor c = cursor_at(g, kv->value_pos);
    return cursor_string(&c, out);
}

const gguf_tensor *gguf_find_tensor(const gguf_file *g, const char *name) {
    if (!g || !name) return NULL;

    const size_t len = strlen(name);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (g->tensors[i].name.len == (uint64_t)len &&
            memcmp(g->tensors[i].name.ptr, name, len) == 0) {
            return &g->tensors[i];
        }
    }
    return NULL;
}

const void *gguf_tensor_data(const gguf_file *g, const gguf_tensor *t) {
    if (!g || !g->map || !t) return NULL;
    if (t->abs_offset > g->size || t->bytes > g->size - t->abs_offset) return NULL;
    return g->map + t->abs_offset;
}

const char *gguf_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}
