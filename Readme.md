# Project 3: Program Optimization with Encryption and Hashing

## Introduction

This project focuses on optimizing a program that performs encryption and hashing, applying concepts learned in computer architecture. The goal is to modify the provided C code to improve its time performance.

## Background

The program involves two main cryptographic operations: encryption and hashing.

### Encryption: ChaCha Stream Cipher

Encryption is a reversible process converting plaintext to ciphertext using an algorithm and a key. This project utilizes the **ChaCha stream cipher**.

ChaCha, designed by Daniel J. Bernstein in 2008, is an evolution of the Salsa20 cipher. It generates a pseudorandom keystream from a 256-bit secret key, a 96-bit nonce (number used once), and a counter. This keystream is then XORed with plaintext to produce ciphertext.

The keystream generation involves:
1.  Initializing a 512-bit state matrix with the key, nonce, counter, and predefined constants (e.g., “expand 32-byte k”).
2.  Performing 20 rounds of mixing using a "quarter-round" function, which involves sequential additions, XORs, and bitwise rotations on grouped state words.
3.  Combining the final state with the initial matrix and serializing it into a 512-bit keystream block.
4.  Incrementing the counter for each subsequent block to ensure uniqueness.

### Hash: Hash Tree (Merkle Tree)

Hashing is a one-way function transforming input data into a fixed-length string of characters (digest). It is irreversible and deterministic. This project uses a **Hash Tree (Merkle Tree)** to process the previously obtained ciphertext.

A Hash Tree (or Merkle tree) is a hierarchical data structure used to efficiently verify the integrity of large datasets.
1.  Leaf nodes represent the hash of a data block.
2.  Non-leaf nodes are hashes of their combined child nodes.
3.  This process continues recursively until a single root hash, known as the Merkle root, is generated.

In this project, we only care about the root node of the Hash Tree and use it as the final output hash value.

### Files

The framework contains the following files and folders:

*   `main.c`: Contains the main function code.
*   `tool.c`: Contains a tool for generating test data.
*   `src/`: Contains encrypt and hash function implementations. 
*   `testcases/`: Contains some test cases. Before using them, you need to generate the corresponding test data with the provided tool.
*   `Makefile`: Used to generate the program and tool. 

### Test Locally

Due to the large size of the input files, test data is not provided directly. You need to generate the relevant data locally.

1.  **Compile Program and Tool:**
    ```bash
    make all
    ```

2.  **Generate Test Data:**
    ```bash
    ./tool ./testcases/test_0.meta
    ```
    This will generate the required data files based on the metadata.

3.  **Test Your Program:**
    ```bash
    ./program ./testcases/test_0.meta
    ```

## Optimization Techniques

*   **Multithreading:**
    *   Use OpenMP or `pthread` to parallelize.
*   **SIMD Instructions:**
    *   SIMD  processes multiple data in parallel.
*   **Loop Unrolling:**
    *   Loop unrolling can be used to reduce the overhead of loops.
*   **Cache Blocking:**
    *   Make memory access more efficient and exploit memory locality.