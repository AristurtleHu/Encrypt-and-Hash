#include "mercha.h"
#include <immintrin.h>

// Helper for ROTL32 on __m256i
#define rotl32_256(x, n)                                                       \
  _mm256_or_si256(_mm256_slli_epi32((x), (n)), _mm256_srli_epi32((x), 32 - (n)))

#define chacha_simd_avx2(a, b, c, d)                                           \
  (a) = _mm256_add_epi32((a), (b));                                            \
  (d) = _mm256_xor_si256((d), (a));                                            \
  (d) = rotl32_256((d), 16);                                                   \
  (c) = _mm256_add_epi32((c), (d));                                            \
  (b) = _mm256_xor_si256((b), (c));                                            \
  (b) = rotl32_256((b), 12);                                                   \
  (a) = _mm256_add_epi32((a), (b));                                            \
  (d) = _mm256_xor_si256((d), (a));                                            \
  (d) = rotl32_256((d), 8);                                                    \
  (c) = _mm256_add_epi32((c), (d));                                            \
  (b) = _mm256_xor_si256((b), (c));                                            \
  (b) = rotl32_256((b), 7)

static void chacha20_double_block_avx2(const uint32_t state_template[16],
                                       uint32_t counter_block0,
                                       uint8_t out[128]) {

  __m256i v_s0_3, v_s4_7, v_s8_11, v_s12_15;
  __m256i v_x0_3, v_x4_7, v_x8_11, v_x12_15;

  uint32_t current_state1[16], current_state2[16];

  memcpy(current_state1, state_template, 16 * sizeof(uint32_t));
  memcpy(current_state2, state_template, 16 * sizeof(uint32_t));

  current_state1[12] = counter_block0;
  current_state2[12] = counter_block0 + 1;

  // Load states: low 128-bits for current_state1, high 128-bits for
  // current_state2
  v_s0_3 =
      _mm256_set_m128i(_mm_loadu_si128((const __m128i *)&current_state2[0]),
                       _mm_loadu_si128((const __m128i *)&current_state1[0]));
  v_s4_7 =
      _mm256_set_m128i(_mm_loadu_si128((const __m128i *)&current_state2[4]),
                       _mm_loadu_si128((const __m128i *)&current_state1[4]));
  v_s8_11 =
      _mm256_set_m128i(_mm_loadu_si128((const __m128i *)&current_state2[8]),
                       _mm_loadu_si128((const __m128i *)&current_state1[8]));
  v_s12_15 =
      _mm256_set_m128i(_mm_loadu_si128((const __m128i *)&current_state2[12]),
                       _mm_loadu_si128((const __m128i *)&current_state1[12]));

  v_x0_3 = v_s0_3;
  v_x4_7 = v_s4_7;
  v_x8_11 = v_s8_11;
  v_x12_15 = v_s12_15;

  // 10 double rounds (20 quarter rounds)
  for (int i = 0; i < 10; ++i) {
    // Column rounds
    chacha_simd_avx2(v_x0_3, v_x4_7, v_x8_11, v_x12_15);
    // Diagonal rounds
    // Shuffle (rotate) elements within each 128-bit lane
    v_x4_7 = _mm256_shuffle_epi32(v_x4_7,
                                  _MM_SHUFFLE(0, 3, 2, 1)); // Rotate left by 1
    v_x8_11 = _mm256_shuffle_epi32(v_x8_11,
                                   _MM_SHUFFLE(1, 0, 3, 2)); // Rotate left by 2
    v_x12_15 = _mm256_shuffle_epi32(
        v_x12_15, _MM_SHUFFLE(2, 1, 0, 3)); // Rotate left by 3

    chacha_simd_avx2(v_x0_3, v_x4_7, v_x8_11, v_x12_15);

    // Shuffle back (rotate right) elements within each 128-bit lane
    v_x4_7 = _mm256_shuffle_epi32(
        v_x4_7, _MM_SHUFFLE(2, 1, 0, 3)); // Rotate right by 1 (left by 3)
    v_x8_11 = _mm256_shuffle_epi32(
        v_x8_11, _MM_SHUFFLE(1, 0, 3, 2)); // Rotate right by 2 (left by 2)
    v_x12_15 = _mm256_shuffle_epi32(
        v_x12_15, _MM_SHUFFLE(0, 3, 2, 1)); // Rotate right by 3 (left by 1)
  }

  // Add initial state
  v_x0_3 = _mm256_add_epi32(v_x0_3, v_s0_3);
  v_x4_7 = _mm256_add_epi32(v_x4_7, v_s4_7);
  v_x8_11 = _mm256_add_epi32(v_x8_11, v_s8_11);
  v_x12_15 = _mm256_add_epi32(v_x12_15, v_s12_15);

  // Store results: block0 keystream, then block1 keystream
  _mm_storeu_si128((__m128i *)(out + 0), _mm256_castsi256_si128(v_x0_3));
  _mm_storeu_si128((__m128i *)(out + 64), _mm256_extracti128_si256(v_x0_3, 1));

  _mm_storeu_si128((__m128i *)(out + 16), _mm256_castsi256_si128(v_x4_7));
  _mm_storeu_si128((__m128i *)(out + 64 + 16),
                   _mm256_extracti128_si256(v_x4_7, 1));

  _mm_storeu_si128((__m128i *)(out + 32), _mm256_castsi256_si128(v_x8_11));
  _mm_storeu_si128((__m128i *)(out + 64 + 32),
                   _mm256_extracti128_si256(v_x8_11, 1));

  _mm_storeu_si128((__m128i *)(out + 48), _mm256_castsi256_si128(v_x12_15));
  _mm_storeu_si128((__m128i *)(out + 64 + 48),
                   _mm256_extracti128_si256(v_x12_15, 1));
}

// Helper function to XOR buffer with keystream
static inline void xor_buffer_simd(uint8_t *buf, const uint8_t *key_stream,
                                   size_t len) {
  size_t i = 0;
  for (; i + 15 < len; i += 16) {
    __m128i b_chunk = _mm_loadu_si128((const __m128i *)(buf + i));
    __m128i k_chunk = _mm_loadu_si128((const __m128i *)(key_stream + i));
    b_chunk = _mm_xor_si128(b_chunk, k_chunk);
    _mm_storeu_si128((__m128i *)(buf + i), b_chunk);
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

  // This is the template state. The counter part (state[12]) will be set per
  // block.
  const uint32_t state_template[16] = {0x61707865,
                                       0x3320646e,
                                       0x79622d32,
                                       0x6b206574,
                                       key_words[0],
                                       key_words[1],
                                       key_words[2],
                                       key_words[3],
                                       key_words[4],
                                       key_words[5],
                                       key_words[6],
                                       key_words[7],
                                       0, // Counter placeholder
                                       nonce_words[0],
                                       nonce_words[1],
                                       nonce_words[2]};

  size_t num_total_blocks = (length + 63) / 64;

#pragma omp parallel for
  for (size_t block_idx_base = 0; block_idx_base < num_total_blocks;
       block_idx_base += 2) {
    uint8_t key_stream_pair[128]; // For two blocks
    uint32_t counter_for_block0 = initial_counter + (uint32_t)block_idx_base;

    chacha20_double_block_avx2(state_template, counter_for_block0,
                               key_stream_pair);

    // XOR for the first block in the pair (block_idx_base)
    size_t offset0 = block_idx_base * 64;
    if (offset0 < length) { // Check if this block is part of the message length
      size_t remaining_length0 = length - offset0;
      size_t bytes_in_block0 =
          (remaining_length0 < 64) ? remaining_length0 : 64;
      xor_buffer_simd(buffer + offset0, key_stream_pair, bytes_in_block0);
    }

    // XOR for the second block in the pair (block_idx_base + 1), if it exists
    if (block_idx_base + 1 < num_total_blocks) {
      size_t offset1 = (block_idx_base + 1) * 64;
      if (offset1 <
          length) { // Check if this block is part of the message length
        size_t remaining_length1 = length - offset1;
        size_t bytes_in_block1 =
            (remaining_length1 < 64) ? remaining_length1 : 64;
        xor_buffer_simd(buffer + offset1, key_stream_pair + 64,
                        bytes_in_block1);
      }
    }
  }
}
