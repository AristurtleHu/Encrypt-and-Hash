#include "mercha.h"
#include <omp.h>

void merge_hash(const uint8_t block1[64], const uint8_t block2[64],
                uint8_t output[64]) {

  uint32_t state[16];

  const uint32_t *w1 = (const uint32_t *)block1;
  const uint32_t *w2 = (const uint32_t *)block2;

  // Fully unrolled loop
  state[0] = w1[0] ^ w2[7];
  state[8] = w2[0] ^ w1[7];
  state[1] = w1[1] ^ w2[6];
  state[9] = w2[1] ^ w1[6];
  state[2] = w1[2] ^ w2[5];
  state[10] = w2[2] ^ w1[5];
  state[3] = w1[3] ^ w2[4];
  state[11] = w2[3] ^ w1[4];
  state[4] = w1[4] ^ w2[3];
  state[12] = w2[4] ^ w1[3];
  state[5] = w1[5] ^ w2[2];
  state[13] = w2[5] ^ w1[2];
  state[6] = w1[6] ^ w2[1];
  state[14] = w2[6] ^ w1[1];
  state[7] = w1[7] ^ w2[0];
  state[15] = w2[7] ^ w1[0];

  for (int round = 0; round < 10; ++round) {
    // Unrolled first inner loop (original i=0 to 3)
    state[0] += state[4];
    state[0] = ROTL32(state[0], 7);
    state[8] += state[12];
    state[8] = ROTL32(state[8], 7);
    state[1] += state[5];
    state[1] = ROTL32(state[1], 7);
    state[9] += state[13];
    state[9] = ROTL32(state[9], 7);
    state[2] += state[6];
    state[2] = ROTL32(state[2], 7);
    state[10] += state[14];
    state[10] = ROTL32(state[10], 7);
    state[3] += state[7];
    state[3] = ROTL32(state[3], 7);
    state[11] += state[15];
    state[11] = ROTL32(state[11], 7);

    // Unrolled second inner loop (original i=0 to 3)
    state[0] += state[8];
    state[0] = ROTL32(state[0], 9);
    state[4] += state[12];
    state[4] = ROTL32(state[4], 9);
    state[1] += state[9];
    state[1] = ROTL32(state[1], 9);
    state[5] += state[13];
    state[5] = ROTL32(state[5], 9);
    state[2] += state[10];
    state[2] = ROTL32(state[2], 9);
    state[6] += state[14];
    state[6] = ROTL32(state[6], 9);
    state[3] += state[11];
    state[3] = ROTL32(state[3], 9);
    state[7] += state[15];
    state[7] = ROTL32(state[7], 9);
  }

  // Fully unrolled final accumulation loop
  state[0] += state[15];
  state[1] += state[14];
  state[2] += state[13];
  state[3] += state[12];
  state[4] += state[11];
  state[5] += state[10];
  state[6] += state[9];
  state[7] += state[8];

  memcpy(output, state, 64);
}

void merkel_tree(const uint8_t *input, uint8_t *output, size_t length) {

  uint8_t *cur_buf = (uint8_t *)malloc(length);
  uint8_t *prev_buf = (uint8_t *)malloc(length);
  memcpy(prev_buf, input, length);

  size_t current_length = length / 2;

  while (current_length >= 64) {
    size_t num = current_length / 64;

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
      merge_hash(prev_buf + (2 * i) * 64, prev_buf + (2 * i + 1) * 64,
                 cur_buf + i * 64);
    }
    current_length /= 2;
    uint8_t *tmp = cur_buf;
    cur_buf = prev_buf;
    prev_buf = tmp;
  }

  memcpy(output, cur_buf, 64);
  free(cur_buf);
  free(prev_buf);
}