#include "mercha.h"
#include <immintrin.h>

// Helper for ROTL32 on __m256i using inline assembly
static inline __m256i rotl32_256(__m256i x, int n) {
  __m256i result;
  __asm__ volatile("vpsrld $%c2, %1, %%ymm0\n\t"
                   "vpslld $%c3, %1, %0\n\t"
                   "vpor %%ymm0, %0, %0"
                   : "=x"(result)
                   : "x"(x), "i"(32 - n), "i"(n)
                   : "ymm0");
  return result;
}

#define chacha_simd(a, b, c, d)                                                \
  __asm__ volatile("vpaddd %1, %0, %0\n\t"                                     \
                   "vpxor %0, %3, %3\n\t"                                      \
                   "vpsrld $16, %3, %%ymm15\n\t"                               \
                   "vpslld $16, %3, %3\n\t"                                    \
                   "vpor %%ymm15, %3, %3\n\t"                                  \
                   "vpaddd %3, %2, %2\n\t"                                     \
                   "vpxor %2, %1, %1\n\t"                                      \
                   "vpsrld $20, %1, %%ymm15\n\t"                               \
                   "vpslld $12, %1, %1\n\t"                                    \
                   "vpor %%ymm15, %1, %1\n\t"                                  \
                   "vpaddd %1, %0, %0\n\t"                                     \
                   "vpxor %0, %3, %3\n\t"                                      \
                   "vpsrld $24, %3, %%ymm15\n\t"                               \
                   "vpslld $8, %3, %3\n\t"                                     \
                   "vpor %%ymm15, %3, %3\n\t"                                  \
                   "vpaddd %3, %2, %2\n\t"                                     \
                   "vpxor %2, %1, %1\n\t"                                      \
                   "vpsrld $25, %1, %%ymm15\n\t"                               \
                   "vpslld $7, %1, %1\n\t"                                     \
                   "vpor %%ymm15, %1, %1"                                      \
                   : "+x"(a), "+x"(b), "+x"(c), "+x"(d)                        \
                   :                                                           \
                   : "ymm15")

#define round                                                                  \
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);                              \
  __asm__ volatile("vpshufd $0x39, %0, %0\n\t"                                 \
                   "vpshufd $0x4e, %1, %1\n\t"                                 \
                   "vpshufd $0x93, %2, %2"                                     \
                   : "+x"(v_x4_7), "+x"(v_x8_11), "+x"(v_x12_15));             \
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);                              \
  __asm__ volatile("vpshufd $0x93, %0, %0\n\t"                                 \
                   "vpshufd $0x4e, %1, %1\n\t"                                 \
                   "vpshufd $0x39, %2, %2"                                     \
                   : "+x"(v_x4_7), "+x"(v_x8_11), "+x"(v_x12_15));

// Assumes state and out are 16-byte aligned.
static void chacha20_block(uint32_t state[16], uint8_t out[128]) {

  __m256i v_s0_3, v_s4_7, v_s8_11, v_s12_15;
  __m256i v_x0_3, v_x4_7, v_x8_11, v_x12_15;

  // Ensure current_state1 and current_state2 are 16-byte aligned for
  // _mm_load_si128
  uint32_t current_state1[16] __attribute__((aligned(16)));
  uint32_t current_state2[16] __attribute__((aligned(16)));

  memcpy(current_state1, state, 16 * sizeof(uint32_t));
  memcpy(current_state2, state, 16 * sizeof(uint32_t));
  current_state2[12] += 1;

  // low 128-bits for current_state1, high 128-bits for current_state2
  v_s0_3 =
      _mm256_set_m128i(_mm_load_si128((const __m128i *)&current_state2[0]),
                       _mm_load_si128((const __m128i *)&current_state1[0]));
  v_s4_7 =
      _mm256_set_m128i(_mm_load_si128((const __m128i *)&current_state2[4]),
                       _mm_load_si128((const __m128i *)&current_state1[4]));
  v_s8_11 =
      _mm256_set_m128i(_mm_load_si128((const __m128i *)&current_state2[8]),
                       _mm_load_si128((const __m128i *)&current_state1[8]));
  v_s12_15 =
      _mm256_set_m128i(_mm_load_si128((const __m128i *)&current_state2[12]),
                       _mm_load_si128((const __m128i *)&current_state1[12]));

  v_x0_3 = v_s0_3;
  v_x4_7 = v_s4_7;
  v_x8_11 = v_s8_11;
  v_x12_15 = v_s12_15;

  // Rounds
  for (int i = 0; i < 10; i += 2) {
    round;
    round;
  }

  // Add initial state
  v_x0_3 = _mm256_add_epi32(v_x0_3, v_s0_3);
  v_x4_7 = _mm256_add_epi32(v_x4_7, v_s4_7);
  v_x8_11 = _mm256_add_epi32(v_x8_11, v_s8_11);
  v_x12_15 = _mm256_add_epi32(v_x12_15, v_s12_15);

  // Store results: block0 keystream, then block1 keystream
  // Assumes out is 16-byte aligned. Offsets are multiples of 16.
  _mm_store_si128((__m128i *)(out + 0), _mm256_castsi256_si128(v_x0_3));
  _mm_store_si128((__m128i *)(out + 64), _mm256_extracti128_si256(v_x0_3, 1));

  _mm_store_si128((__m128i *)(out + 16), _mm256_castsi256_si128(v_x4_7));
  _mm_store_si128((__m128i *)(out + 64 + 16),
                  _mm256_extracti128_si256(v_x4_7, 1));

  _mm_store_si128((__m128i *)(out + 32), _mm256_castsi256_si128(v_x8_11));
  _mm_store_si128((__m128i *)(out + 64 + 32),
                  _mm256_extracti128_si256(v_x8_11, 1));

  _mm_store_si128((__m128i *)(out + 48), _mm256_castsi256_si128(v_x12_15));
  _mm_store_si128((__m128i *)(out + 64 + 48),
                  _mm256_extracti128_si256(v_x12_15, 1));
}

// Helper function to XOR buffer with keystream
static inline void xor_buffer_simd(uint8_t *buf, const uint8_t *key_stream,
                                   size_t len) {
  size_t i = 0;
  for (; i + 15 < len; i += 16) {
    __m128i b_chunk = _mm_load_si128((const __m128i *)(buf + i));
    __m128i k_chunk = _mm_load_si128((const __m128i *)(key_stream + i));
    b_chunk = _mm_xor_si128(b_chunk, k_chunk);
    _mm_store_si128((__m128i *)(buf + i), b_chunk);
  }

  for (; i < len; ++i) {
    buf[i] ^= key_stream[i];
  }
}

void chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                      uint32_t initial_counter, uint8_t *buffer,
                      size_t length) {

  uint32_t key_words[8];
  uint32_t nonce_words[3];

  memcpy(key_words, key, 32);
  memcpy(nonce_words, nonce, 12);

  uint32_t state[16] __attribute__((aligned(16))) = {
      0x61707865,      0x3320646e,     0x79622d32,     0x6b206574,
      key_words[0],    key_words[1],   key_words[2],   key_words[3],
      key_words[4],    key_words[5],   key_words[6],   key_words[7],
      initial_counter, nonce_words[0], nonce_words[1], nonce_words[2]};

  size_t num_blocks = (length + 63) / 64; // Total 64-byte ChaCha blocks needed

#pragma omp parallel for
  for (size_t pair_idx = 0; pair_idx < (num_blocks + 1) / 2; ++pair_idx) {
    size_t block_idx = pair_idx * 2;

    uint8_t key_stream_pair[128] __attribute__((aligned(16)));
    uint32_t local_state[16] __attribute__((aligned(16)));

    memcpy(local_state, state, sizeof(state));
    local_state[12] += (uint32_t)block_idx;

    chacha20_block(local_state, key_stream_pair);

    // XOR for the first block in the pair
    size_t offset0 = block_idx * 64;
    if (offset0 < length) {
      size_t remaining = length - offset0;
      size_t bytes_to_xor = (remaining < 64) ? remaining : 64;
      xor_buffer_simd(buffer + offset0, key_stream_pair, bytes_to_xor);
    }

    // XOR for the second block in the pair
    size_t block_idx_for_second = block_idx + 1;
    if (block_idx_for_second < num_blocks) {
      size_t offset1 = block_idx_for_second * 64;
      if (offset1 < length) {
        size_t remaining = length - offset1;
        size_t bytes_to_xor = (remaining < 64) ? remaining : 64;
        xor_buffer_simd(buffer + offset1, key_stream_pair + 64, bytes_to_xor);
      }
    }
  }
}
