#include "mercha.h"
#include <immintrin.h>

// Helper for ROTL32 on __m128i 4int
#define rotl32_128(x, n)                                                       \
  _mm_or_si128(_mm_slli_epi32((x), (n)), _mm_srli_epi32((x), 32 - (n)))

#define round                                                                  \
  v_state0_3 = _mm_add_epi32(v_state0_3, v_state4_7);                          \
  v_state0_3 = rotl32_128(v_state0_3, 7);                                      \
                                                                               \
  v_state8_11 = _mm_add_epi32(v_state8_11, v_state12_15);                      \
  v_state8_11 = rotl32_128(v_state8_11, 7);                                    \
                                                                               \
  v_state0_3 = _mm_add_epi32(v_state0_3, v_state8_11);                         \
  v_state0_3 = rotl32_128(v_state0_3, 9);                                      \
                                                                               \
  v_state4_7 = _mm_add_epi32(v_state4_7, v_state12_15);                        \
  v_state4_7 = rotl32_128(v_state4_7, 9);

void merge_hash(const uint8_t block1[64], const uint8_t block2[64],
                uint8_t output[64]) {
  __m256i v_state0_7, v_state8_15;

  __m256i v_w1 = _mm256_loadu_si256((const __m256i *)block1);
  __m256i v_w2 = _mm256_loadu_si256((const __m256i *)block2);

  // Control vector for reversing elements in a __m256i vector
  const __m256i v_reverse_control = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

  // state[i] = w1[i] ^ w2[7 - i];
  __m256i v_w2_rev = _mm256_permutevar8x32_epi32(v_w2, v_reverse_control);
  v_state0_7 = _mm256_xor_si256(v_w1, v_w2_rev);

  // state[8 + i] = w2[i] ^ w1[7 - i];
  __m256i v_w1_rev = _mm256_permutevar8x32_epi32(v_w1, v_reverse_control);
  v_state8_15 = _mm256_xor_si256(v_w2, v_w1_rev);

  // Rounds
  __m128i v_state0_3 = _mm256_castsi256_si128(v_state0_7);
  __m128i v_state4_7 = _mm256_extracti128_si256(v_state0_7, 1);
  __m128i v_state8_11 = _mm256_castsi256_si128(v_state8_15);
  __m128i v_state12_15 = _mm256_extracti128_si256(v_state8_15, 1);

  v_state0_3 = _mm_add_epi32(v_state0_3, v_state4_7);
  v_state0_3 = rotl32_128(v_state0_3, 7);

  v_state8_11 = _mm_add_epi32(v_state8_11, v_state12_15);
  v_state8_11 = rotl32_128(v_state8_11, 7);

  v_state0_3 = _mm_add_epi32(v_state0_3, v_state8_11);
  v_state0_3 = rotl32_128(v_state0_3, 9);

  v_state4_7 = _mm_add_epi32(v_state4_7, v_state12_15);
  v_state4_7 = rotl32_128(v_state4_7, 9);

  round;
  round;
  round;
  round;
  round;
  round;
  round;
  round;
  round;

  v_state0_7 = _mm256_set_m128i(v_state4_7, v_state0_3);
  v_state8_15 = _mm256_set_m128i(v_state12_15, v_state8_11);
  // Final summation: state[i] += state[15 - i];
  __m256i v_state8_15_rev =
      _mm256_permutevar8x32_epi32(v_state8_15, v_reverse_control);
  v_state0_7 = _mm256_add_epi32(v_state0_7, v_state8_15_rev);

  // Store result
  _mm256_storeu_si256((__m256i *)output, v_state0_7);
  _mm256_storeu_si256((__m256i *)(output + 32), v_state8_15);
}

void merkel_tree(const uint8_t *input, uint8_t *output, size_t length) {
  uint8_t *cur_buf = (uint8_t *)_mm_malloc(length, 32);
  uint8_t *prev_buf = (uint8_t *)_mm_malloc(length, 32);

  memcpy(prev_buf, input, length);

  length /= 2;
  while (length >= 64) {
    size_t k;
    const size_t prefetch_stride = 64;
    const size_t unroll_factor = 4; // Unroll 4 prefetches per loop iteration
    const size_t unroll_step = unroll_factor * prefetch_stride;

    size_t limit_prev = 2 * length;
    for (k = 0; k < limit_prev / unroll_step * unroll_step; k += unroll_step) {
      _mm_prefetch((const char *)(prev_buf + k), _MM_HINT_T0);
      _mm_prefetch((const char *)(prev_buf + k + prefetch_stride), _MM_HINT_T0);
      _mm_prefetch((const char *)(prev_buf + k + 2 * prefetch_stride),
                   _MM_HINT_T0);
      _mm_prefetch((const char *)(prev_buf + k + 3 * prefetch_stride),
                   _MM_HINT_T0);
    }
    for (; k < limit_prev; k += prefetch_stride) {
      _mm_prefetch((const char *)(prev_buf + k), _MM_HINT_T0);
    }

    size_t num_output_blocks =
        length / 64; // Total number of merge_hash operations for this level

    if (num_output_blocks >= 32) {
#pragma omp parallel for
      for (size_t i = 0; i < num_output_blocks / 2; ++i) {
        merge_hash(prev_buf + (4 * i) * 64, prev_buf + (4 * i + 1) * 64,
                   cur_buf + (2 * i) * 64);

        merge_hash(prev_buf + (4 * i + 2) * 64, prev_buf + (4 * i + 3) * 64,
                   cur_buf + (2 * i + 1) * 64);
      }
    } else {
      for (size_t i = 0; i < num_output_blocks / 2; ++i) {
        merge_hash(prev_buf + (4 * i) * 64, prev_buf + (4 * i + 1) * 64,
                   cur_buf + (2 * i) * 64);

        merge_hash(prev_buf + (4 * i + 2) * 64, prev_buf + (4 * i + 3) * 64,
                   cur_buf + (2 * i + 1) * 64);
      }
    }

    // If the total number of merge operations is odd, handle the last one
    if (num_output_blocks % 2 != 0) {
      size_t last_output_idx = num_output_blocks - 1;
      merge_hash(prev_buf + (2 * last_output_idx) * 64,
                 prev_buf + (2 * last_output_idx + 1) * 64,
                 cur_buf + last_output_idx * 64);
    }

    length /= 2;
    uint8_t *tmp = cur_buf;
    cur_buf = prev_buf;
    prev_buf = tmp;
  }

  memcpy(output, prev_buf, 64);
  _mm_free(cur_buf);
  _mm_free(prev_buf);
}