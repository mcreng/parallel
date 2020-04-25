/*
 * Do not change this file.
 * This is a CUDA version of the smith waterman algorithm
 * Compile: nvcc -std=c++11 -arch=compute_52 -code=sm_52 main.cu cuda_smith_waterman_skeleton.cu -o cuda_smith_waterman
 * Run: ./cuda_smith_waterman <input file> <num of blocks per grid> <number of thread per block>
 */

#pragma once

const int MATCH = 3, MIS = -3, GAP = 2;


int smith_waterman(int blocks_per_grid, int threads_per_block, char *a, char *b, int a_len, int b_len);

inline __device__ int sub_mat(char x, char y) {
	return x == y ? MATCH : MIS;
}

#define GPUErrChk(ans) { utils::GPUAssert((ans), __FILE__, __LINE__); }

namespace utils {

	inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
		if (code != cudaSuccess) {
			fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort)
				exit(code);
		}
	}

	inline __device__ int dev_idx(int x, int y, int n) {
		return x * n + y;
	}
}
