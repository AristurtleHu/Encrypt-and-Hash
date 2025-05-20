#include "mercha.h"
#include <immintrin.h>

// Helper for ROTL32 on __m128i 4int
#define rotl32_128(x, n)                                                       \
  _mm_or_si128(_mm_slli_epi32((x), (n)), _mm_srli_epi32((x), 32 - (n)))

#define chacha_simd(a, b, c, d)                                                \
  (a) = _mm_add_epi32((a), (b));                                               \
  (d) = _mm_xor_si128((d), (a));                                               \
  (d) = rotl32_128((d), 16);                                                   \
  (c) = _mm_add_epi32((c), (d));                                               \
  (b) = _mm_xor_si128((b), (c));                                               \
  (b) = rotl32_128((b), 12);                                                   \
  (a) = _mm_add_epi32((a), (b));                                               \
  (d) = _mm_xor_si128((d), (a));                                               \
  (d) = rotl32_128((d), 8);                                                    \
  (c) = _mm_add_epi32((c), (d));                                               \
  (b) = _mm_xor_si128((b), (c));                                               \
  (b) = rotl32_128((b), 7)

static void chacha20_block(uint32_t state[16], uint8_t out[64]) {

  __m128i v_s0_3, v_s4_7, v_s8_11, v_s12_15;
  __m128i v_x0_3, v_x4_7, v_x8_11, v_x12_15;

  // State
  v_s0_3 = _mm_loadu_si128((const __m128i *)&state[0]);
  v_s4_7 = _mm_loadu_si128((const __m128i *)&state[4]);
  v_s8_11 = _mm_loadu_si128((const __m128i *)&state[8]);
  v_s12_15 = _mm_loadu_si128((const __m128i *)&state[12]);

  // X
  v_x0_3 = v_s0_3;
  v_x4_7 = v_s4_7;
  v_x8_11 = v_s8_11;
  v_x12_15 = v_s12_15;

  // Rounds

  // Round 1
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 2
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 3
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 4
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 5
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 6
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 7
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 8
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 9
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Round 10
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(0, 3, 2, 1));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(2, 1, 0, 3));
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
  v_x4_7 = _mm_shuffle_epi32(v_x4_7, _MM_SHUFFLE(2, 1, 0, 3));
  v_x8_11 = _mm_shuffle_epi32(v_x8_11, _MM_SHUFFLE(1, 0, 3, 2));
  v_x12_15 = _mm_shuffle_epi32(v_x12_15, _MM_SHUFFLE(0, 3, 2, 1));

  // Add initial state
  v_x0_3 = _mm_add_epi32(v_x0_3, v_s0_3);
  v_x4_7 = _mm_add_epi32(v_x4_7, v_s4_7);
  v_x8_11 = _mm_add_epi32(v_x8_11, v_s8_11);
  v_x12_15 = _mm_add_epi32(v_x12_15, v_s12_15);

  // Store results
  _mm_storeu_si128((__m128i *)(out + 0), v_x0_3);
  _mm_storeu_si128((__m128i *)(out + 16), v_x4_7);
  _mm_storeu_si128((__m128i *)(out + 32), v_x8_11);
  _mm_storeu_si128((__m128i *)(out + 48), v_x12_15);
}

void chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                      uint32_t initial_counter, uint8_t *buffer,
                      size_t length) {

  uint32_t key_words[8];
  uint32_t nonce_words[3];

  memcpy(key_words, key, 32);
  memcpy(nonce_words, nonce, 12);

  uint32_t state[16] = {
      0x61707865,      0x3320646e,     0x79622d32,     0x6b206574,
      key_words[0],    key_words[1],   key_words[2],   key_words[3],
      key_words[4],    key_words[5],   key_words[6],   key_words[7],
      initial_counter, nonce_words[0], nonce_words[1], nonce_words[2]};

  size_t num_blocks = (length + 63) / 64;

#pragma omp parallel for
  for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    uint32_t local_state[16];
    memcpy(local_state, state, sizeof(state));
    local_state[12] = initial_counter + (uint32_t)block_idx;

    uint8_t key_stream[64];
    chacha20_block(local_state, key_stream);

    size_t current_block_offset = block_idx * 64;
    size_t remaining_length = length - current_block_offset;
    size_t bytes_in_block = (remaining_length < 64) ? remaining_length : 64;

    uint8_t *buf_ptr = buffer + current_block_offset;
    size_t i = 0;
    for (; i + 15 < bytes_in_block; i += 16) {
      __m128i b_chunk = _mm_loadu_si128((const __m128i *)(buf_ptr + i));
      __m128i k_chunk = _mm_loadu_si128((const __m128i *)(key_stream + i));
      b_chunk = _mm_xor_si128(b_chunk, k_chunk);
      _mm_storeu_si128((__m128i *)(buf_ptr + i), b_chunk);
    }
    for (; i < bytes_in_block; ++i) {
      buf_ptr[i] ^= key_stream[i];
    }
  }
}