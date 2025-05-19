#include "mercha.h"

void merge_hash(const uint8_t block1[64], const uint8_t block2[64],
                uint8_t output[64]) {

  uint32_t state[16] = {0};

  const uint32_t *w1 = (const uint32_t *)block1;
  const uint32_t *w2 = (const uint32_t *)block2;

  for (int i = 0; i < 8; ++i) {
    state[i] = w1[i] ^ w2[7 - i];
    state[8 + i] = w2[i] ^ w1[7 - i];
  }

  for (int round = 0; round < 10; ++round) {
    for (int i = 0; i < 4; ++i) {
      state[i] += state[4 + i];
      state[i] = ROTL32(state[i], 7);
      state[8 + i] += state[12 + i];
      state[8 + i] = ROTL32(state[8 + i], 7);
    }
    for (int i = 0; i < 4; ++i) {
      state[i] += state[8 + i];
      state[i] = ROTL32(state[i], 9);
      state[4 + i] += state[12 + i];
      state[4 + i] = ROTL32(state[4 + i], 9);
    }
  }

  for (int i = 0; i < 8; ++i) {
    state[i] += state[15 - i];
  }

  memcpy(output, state, 64);
}

void merkel_tree(const uint8_t *input, uint8_t *output, size_t length) {

  uint8_t *cur_buf = malloc(length);
  uint8_t *prev_buf = malloc(length);
  memcpy(prev_buf, input, length);

  length /= 2;
  while (length >= 64) {
    for (int i = 0; i < length / 64; ++i) {
      merge_hash(prev_buf + (2 * i) * 64, prev_buf + (2 * i + 1) * 64,
                 cur_buf + i * 64);
    }
    length /= 2;
    uint8_t *tmp = cur_buf;
    cur_buf = prev_buf;
    prev_buf = tmp;
  }

  memcpy(output, cur_buf, 64);
  free(cur_buf);
  free(prev_buf);
}