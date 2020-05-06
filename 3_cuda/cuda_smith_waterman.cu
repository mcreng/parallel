#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <math.h>

using namespace std;

#include "cuda_smith_waterman.h"

__global__ void CalcScoreKernel(int *d_score_t2, int *d_score_t1, int *d_score_t0, char *a, char *b, int *d_a_len, int *d_b_len, int diag, int *d_block_score) {
  extern __shared__ int maxes[];

  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int diag_size = min(*d_a_len, *d_b_len) + 1;
  int cur_max = 0;
  
  for (int tx=tid; tx < diag_size; tx += nthreads) {
    int y = max(0, diag-*d_a_len) + tx;
    int x = diag - y;
  
    if (x > 0 && y > 0 && x <= *d_a_len && y <= *d_b_len) {
      if (diag <= *d_a_len) {
        d_score_t0[tx] = max(0,
                          max(d_score_t1[tx - 1] - GAP,
                          max(d_score_t1[tx] - GAP,
                              d_score_t2[tx - 1] + sub_mat(a[x-1], b[y-1]))));
      } else if (diag == *d_a_len+1) {
        d_score_t0[tx] = max(0,
                          max(d_score_t1[tx] - GAP,
                          max(d_score_t1[tx + 1] - GAP,
                              d_score_t2[tx] + sub_mat(a[x-1], b[y-1]))));
      } else {
        d_score_t0[tx] = max(0,
                          max(d_score_t1[tx] - GAP,
                          max(d_score_t1[tx + 1] - GAP,
                              d_score_t2[tx + 1] + sub_mat(a[x-1], b[y-1]))));
      }
    } else {
      d_score_t0[tx] = 0;
    }
    cur_max = max(cur_max, d_score_t0[tx]);
  }
  maxes[threadIdx.x] = cur_max;

  for (int offset=blockDim.x/2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      maxes[threadIdx.x] = max(maxes[threadIdx.x], maxes[threadIdx.x + offset]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) d_block_score[blockIdx.x] = max(d_block_score[blockIdx.x], maxes[0]);

}

int smith_waterman(int blocks_per_grid, int threads_per_block, char *a, char *b, int a_len, int b_len) {
  
  int *d_score_t2, *d_score_t1, *d_score_t0, *tmp; // score matrix in (diag, idx) coordinates
  int *d_a_len, *d_b_len;
  char *d_a, *d_b;
  size_t size_score = (min(a_len, b_len) + 1) * sizeof(int);

	cudaMalloc(&d_score_t2, size_score);
	cudaMalloc(&d_score_t1, size_score);
  cudaMalloc(&d_score_t0, size_score);
	cudaMalloc(&d_a, a_len * sizeof(int));
  cudaMalloc(&d_b, b_len * sizeof(int));
  cudaMalloc(&d_a_len, sizeof(int));
  cudaMalloc(&d_b_len, sizeof(int));

  cudaMemset(d_score_t2, 0, size_score);
  cudaMemset(d_score_t1, 0, size_score);
  cudaMemset(d_score_t0, 0, size_score);
	cudaMemcpy(d_a, a, a_len*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_len*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_len, &a_len, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_len, &b_len, sizeof(int), cudaMemcpyHostToDevice);

  size_t size_block_score = blocks_per_grid * sizeof(int);
  int *h_block_score, *d_block_score;
  h_block_score = (int *) malloc(size_block_score);
  cudaMalloc(&d_block_score, size_block_score);

  for (int diag=2; diag < a_len + b_len + 1; ++diag) {
    CalcScoreKernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(int)>>>(d_score_t2, d_score_t1, d_score_t0, d_a, d_b, d_a_len, d_b_len, diag, d_block_score);
    tmp = d_score_t2;
    d_score_t2 = d_score_t1;
    d_score_t1 = d_score_t0;
    d_score_t0 = tmp;
  }

  cudaMemcpy(h_block_score, d_block_score, size_block_score, cudaMemcpyDeviceToHost);

  int max = *(std::max_element(h_block_score, h_block_score + blocks_per_grid));

  free(h_block_score);
	cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_block_score);
  cudaFree(d_a_len);
  cudaFree(d_b_len);

  return max;
}