#pragma once

#include "qemu/osdep.h"

#if defined(__x86_64__)
#include <immintrin.h>

static int nyx_nt_copy_enabled = -1;

static inline bool nyx_nt_copy_available(void)
{
    if (nyx_nt_copy_enabled == -1) {
        nyx_nt_copy_enabled = __builtin_cpu_supports("avx2") ? 1 : 0;
        if (getenv("NYX_DISABLE_NT_COPY")) {
            nyx_nt_copy_enabled = 0;
        }
    }
    return nyx_nt_copy_enabled == 1;
}

/*
 * Copy one page with non-temporal stores, skipping the read-for-ownership
 * that regular stores incur. Both pointers are page-aligned (region ptrs
 * come from mmap() / RAMBlock->host, offsets are page offsets), which
 * satisfies the 32-byte alignment _mm256_stream_si256 requires.
 * No sfence here: the caller fences once after the whole restore loop.
 */
__attribute__((target("avx2"), unused))
static void nyx_nt_copy_page(void *dst, const void *src)
{
    __m256i       *d = (__m256i *)dst;
    const __m256i *s = (const __m256i *)src;

    for (int i = 0; i < TARGET_PAGE_SIZE / 32; i += 4) {
        __m256i a = _mm256_load_si256(&s[i + 0]);
        __m256i b = _mm256_load_si256(&s[i + 1]);
        __m256i c = _mm256_load_si256(&s[i + 2]);
        __m256i e = _mm256_load_si256(&s[i + 3]);
        _mm256_stream_si256(&d[i + 0], a);
        _mm256_stream_si256(&d[i + 1], b);
        _mm256_stream_si256(&d[i + 2], c);
        _mm256_stream_si256(&d[i + 3], e);
    }
}

static inline void nyx_restore_copy_page(void *dst, const void *src)
{
    if (nyx_nt_copy_available()) {
        nyx_nt_copy_page(dst, src);
    } else {
        memcpy(dst, src, TARGET_PAGE_SIZE);
    }
}

static inline void nyx_restore_copy_fence(void)
{
    if (nyx_nt_copy_available()) {
        _mm_sfence();
    }
}
#else
static inline void nyx_restore_copy_page(void *dst, const void *src)
{
    memcpy(dst, src, TARGET_PAGE_SIZE);
}
static inline void nyx_restore_copy_fence(void) {}
#endif
