#include "mercha.h"

static void chacha_quarter_round(uint32_t x[16], size_t a, size_t b, size_t c,
                                 size_t d) {
  x[a] = x[a] + x[b];
  x[d] = ROTL32(x[d] ^ x[a], 16);
  x[c] = x[c] + x[d];
  x[b] = ROTL32(x[b] ^ x[c], 12);
  x[a] = x[a] + x[b];
  x[d] = ROTL32(x[d] ^ x[a], 8);
  x[c] = x[c] + x[d];
  x[b] = ROTL32(x[b] ^ x[c], 7);
}

static void chacha20_block(uint32_t state[16], uint8_t out[64]) {

  uint32_t working_state[16];
  memcpy(working_state, state, 64);

  for (int i = 0; i < 10; i++) {
    chacha_quarter_round(working_state, 0, 4, 8, 12);
    chacha_quarter_round(working_state, 1, 5, 9, 13);
    chacha_quarter_round(working_state, 2, 6, 10, 14);
    chacha_quarter_round(working_state, 3, 7, 11, 15);

    chacha_quarter_round(working_state, 0, 5, 10, 15);
    chacha_quarter_round(working_state, 1, 6, 11, 12);
    chacha_quarter_round(working_state, 2, 7, 8, 13);
    chacha_quarter_round(working_state, 3, 4, 9, 14);
  }

  for (int i = 0; i < 16; i++) {
    working_state[i] += state[i];
  }

  for (int i = 0; i < 16; i++) {
    out[i * 4 + 0] = (uint8_t)(working_state[i] >> 0);
    out[i * 4 + 1] = (uint8_t)(working_state[i] >> 8);
    out[i * 4 + 2] = (uint8_t)(working_state[i] >> 16);
    out[i * 4 + 3] = (uint8_t)(working_state[i] >> 24);
  }
}

void chacha20_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                      uint32_t initial_counter, uint8_t *buffer,
                      size_t length) {
  uint32_t key_words[8];
  uint32_t nonce_words[3];

  for (int i = 0; i < 8; i++) {
    key_words[i] = (uint32_t)key[i * 4 + 0] | ((uint32_t)key[i * 4 + 1] << 8) |
                   ((uint32_t)key[i * 4 + 2] << 16) |
                   ((uint32_t)key[i * 4 + 3] << 24);
  }

  for (int i = 0; i < 3; i++) {
    nonce_words[i] =
        (uint32_t)nonce[i * 4 + 0] | ((uint32_t)nonce[i * 4 + 1] << 8) |
        ((uint32_t)nonce[i * 4 + 2] << 16) | ((uint32_t)nonce[i * 4 + 3] << 24);
  }

  uint32_t state[16] = {
      0x61707865,      0x3320646e,     0x79622d32,     0x6b206574,
      key_words[0],    key_words[1],   key_words[2],   key_words[3],
      key_words[4],    key_words[5],   key_words[6],   key_words[7],
      initial_counter, nonce_words[0], nonce_words[1], nonce_words[2]};

  uint8_t key_stream[64];
  size_t offset = 0;

  while (length > 0) {

    chacha20_block(state, key_stream);
    size_t block_size = length < 64 ? length : 64;
    for (size_t i = 0; i < block_size; i++) {
      buffer[offset + i] = buffer[offset + i] ^ key_stream[i];
    }

    offset += block_size;
    length -= block_size;
    state[12]++;
  }
}