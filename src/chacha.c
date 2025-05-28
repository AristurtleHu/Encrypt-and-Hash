#include "mercha.h"
#include <immintrin.h>

#define chacha_simd(a, b, c, d)                                                \
  __asm__ volatile("vpaddd %1, %0, %0\n\t"                                     \
                   "vpxord %0, %3, %3\n\t"                                     \
                   "vprord $16, %3, %3\n\t"                                    \
                   "vpaddd %3, %2, %2\n\t"                                     \
                   "vpxord %2, %1, %1\n\t"                                     \
                   "vprord $20, %1, %1\n\t"                                    \
                   "vpaddd %1, %0, %0\n\t"                                     \
                   "vpxord %0, %3, %3\n\t"                                     \
                   "vprord $24, %3, %3\n\t"                                    \
                   "vpaddd %3, %2, %2\n\t"                                     \
                   "vpxord %2, %1, %1\n\t"                                     \
                   "vprord $25, %1, %1"                                        \
                   : "+v"(a), "+v"(b), "+v"(c), "+v"(d)                        \
                   :                                                           \
                   :)

#define round                                                                  \
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);                              \
  __asm__ volatile("vpshufd $0x39, %0, %0\n\t"                                 \
                   "vpshufd $0x4e, %1, %1\n\t"                                 \
                   "vpshufd $0x93, %2, %2"                                     \
                   : "+v"(v_x4_7), "+v"(v_x8_11), "+v"(v_x12_15));             \
  chacha_simd(v_x0_3, v_x4_7, v_x8_11, v_x12_15);                              \
  __asm__ volatile("vpshufd $0x93, %0, %0\n\t"                                 \
                   "vpshufd $0x4e, %1, %1\n\t"                                 \
                   "vpshufd $0x39, %2, %2"                                     \
                   : "+v"(v_x4_7), "+v"(v_x8_11), "+v"(v_x12_15));

// Assumes state and out are 16-byte aligned. Generates 4 blocks
static void chacha20_block(uint32_t state[16], uint8_t out[256]) {

  // Ensure current states are 16-byte aligned
  uint32_t current_state1[16] __attribute__((aligned(16)));
  uint32_t current_state2[16] __attribute__((aligned(16)));
  uint32_t current_state3[16] __attribute__((aligned(16)));
  uint32_t current_state4[16] __attribute__((aligned(16)));

  memcpy(current_state1, state, 16 * sizeof(uint32_t));
  memcpy(current_state2, state, 16 * sizeof(uint32_t));
  memcpy(current_state3, state, 16 * sizeof(uint32_t));
  memcpy(current_state4, state, 16 * sizeof(uint32_t));

  current_state2[12] += 1;
  current_state3[12] += 2;
  current_state4[12] += 3;

  __m512i v_s0_3, v_s4_7, v_s8_11, v_s12_15;
  __m512i v_x0_3, v_x4_7, v_x8_11, v_x12_15;

  // Load states
  v_s0_3 =
      _mm512_set_epi32(current_state4[3], current_state4[2], current_state4[1],
                       current_state4[0], current_state3[3], current_state3[2],
                       current_state3[1], current_state3[0], current_state2[3],
                       current_state2[2], current_state2[1], current_state2[0],
                       current_state1[3], current_state1[2], current_state1[1],
                       current_state1[0]);

  v_s4_7 =
      _mm512_set_epi32(current_state4[7], current_state4[6], current_state4[5],
                       current_state4[4], current_state3[7], current_state3[6],
                       current_state3[5], current_state3[4], current_state2[7],
                       current_state2[6], current_state2[5], current_state2[4],
                       current_state1[7], current_state1[6], current_state1[5],
                       current_state1[4]);

  v_s8_11 = _mm512_set_epi32(
      current_state4[11], current_state4[10], current_state4[9],
      current_state4[8], current_state3[11], current_state3[10],
      current_state3[9], current_state3[8], current_state2[11],
      current_state2[10], current_state2[9], current_state2[8],
      current_state1[11], current_state1[10], current_state1[9],
      current_state1[8]);

  v_s12_15 = _mm512_set_epi32(
      current_state4[15], current_state4[14], current_state4[13],
      current_state4[12], current_state3[15], current_state3[14],
      current_state3[13], current_state3[12], current_state2[15],
      current_state2[14], current_state2[13], current_state2[12],
      current_state1[15], current_state1[14], current_state1[13],
      current_state1[12]);

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
  v_x0_3 = _mm512_add_epi32(v_x0_3, v_s0_3);
  v_x4_7 = _mm512_add_epi32(v_x4_7, v_s4_7);
  v_x8_11 = _mm512_add_epi32(v_x8_11, v_s8_11);
  v_x12_15 = _mm512_add_epi32(v_x12_15, v_s12_15);

  // Store results for all 4 blocks
  _mm_store_si128((__m128i *)(out + 0), _mm512_extracti32x4_epi32(v_x0_3, 0));
  _mm_store_si128((__m128i *)(out + 64), _mm512_extracti32x4_epi32(v_x0_3, 1));
  _mm_store_si128((__m128i *)(out + 128), _mm512_extracti32x4_epi32(v_x0_3, 2));
  _mm_store_si128((__m128i *)(out + 192), _mm512_extracti32x4_epi32(v_x0_3, 3));

  _mm_store_si128((__m128i *)(out + 16), _mm512_extracti32x4_epi32(v_x4_7, 0));
  _mm_store_si128((__m128i *)(out + 64 + 16),
                  _mm512_extracti32x4_epi32(v_x4_7, 1));
  _mm_store_si128((__m128i *)(out + 128 + 16),
                  _mm512_extracti32x4_epi32(v_x4_7, 2));
  _mm_store_si128((__m128i *)(out + 192 + 16),
                  _mm512_extracti32x4_epi32(v_x4_7, 3));

  _mm_store_si128((__m128i *)(out + 32), _mm512_extracti32x4_epi32(v_x8_11, 0));
  _mm_store_si128((__m128i *)(out + 64 + 32),
                  _mm512_extracti32x4_epi32(v_x8_11, 1));
  _mm_store_si128((__m128i *)(out + 128 + 32),
                  _mm512_extracti32x4_epi32(v_x8_11, 2));
  _mm_store_si128((__m128i *)(out + 192 + 32),
                  _mm512_extracti32x4_epi32(v_x8_11, 3));

  _mm_store_si128((__m128i *)(out + 48),
                  _mm512_extracti32x4_epi32(v_x12_15, 0));
  _mm_store_si128((__m128i *)(out + 64 + 48),
                  _mm512_extracti32x4_epi32(v_x12_15, 1));
  _mm_store_si128((__m128i *)(out + 128 + 48),
                  _mm512_extracti32x4_epi32(v_x12_15, 2));
  _mm_store_si128((__m128i *)(out + 192 + 48),
                  _mm512_extracti32x4_epi32(v_x12_15, 3));
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
  for (size_t quad_idx = 0; quad_idx < (num_blocks + 3) / 4; ++quad_idx) {
    size_t block_idx = quad_idx * 4;

    uint8_t key_stream[256] __attribute__((aligned(16)));
    uint32_t local_state[16] __attribute__((aligned(16)));

    memcpy(local_state, state, sizeof(state));
    local_state[12] += (uint32_t)block_idx;

    chacha20_block(local_state, key_stream);

    // XOR for all four blocks in the quad
    for (int i = 0; i < 4; ++i) {
      size_t current_block_idx = block_idx + i;
      if (current_block_idx < num_blocks) {
        size_t offset = current_block_idx * 64;
        if (offset < length) {
          size_t remaining = length - offset;
          size_t bytes_to_xor = (remaining < 64) ? remaining : 64;
          xor_buffer_simd(buffer + offset, key_stream + (i * 64), bytes_to_xor);
        }
      }
    }
  }
}
