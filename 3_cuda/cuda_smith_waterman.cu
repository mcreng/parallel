#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <math.h>

using namespace std;

#include "cuda_smith_waterman.h"

__global__ void CalcScoreKernel(int *d_score, char *a, char *b, int a_len, int b_len, int diag, int *d_block_score) {
  extern __shared__ int maxes[];

  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int nthreads = blockDim.x * gridDim.x;
  const int diag_size = min(a_len, b_len) + 1;
  // const int n_diags = a_len + b_len + 1;
  int cur_max = 0;
  
  for (int tx=tid; tx < diag_size; tx += nthreads) {
    int y = max(0, diag-a_len) + tx;
    int x = diag - y;
  
    if (x > 0 && y > 0 && x <= a_len && y <= b_len) {
      if (diag <= a_len) {
        d_score[diag_size*diag + tx] = max(0,
                                            max(d_score[diag_size*(diag-1) + tx - 1] - GAP,
                                            max(d_score[diag_size*(diag-1) + tx] - GAP,
                                                d_score[diag_size*(diag-2) + tx - 1] + sub_mat(a[x-1], b[y-1]))));
      } else if (diag == a_len+1) {
        d_score[diag_size*diag + tx] = max(0,
                                            max(d_score[diag_size*(diag-1) + tx] - GAP,
                                            max(d_score[diag_size*(diag-1) + tx + 1] - GAP,
                                                d_score[diag_size*(diag-2) + tx] + sub_mat(a[x-1], b[y-1]))));
      } else {
        d_score[diag_size*diag + tx] = max(0,
                                            max(d_score[diag_size*(diag-1) + tx] - GAP,
                                            max(d_score[diag_size*(diag-1) + tx + 1] - GAP,
                                                d_score[diag_size*(diag-2) + tx + 1] + sub_mat(a[x-1], b[y-1]))));
      }
    } else {
      d_score[diag_size*diag + tx] = 0;
    }
    cur_max = max(cur_max, d_score[diag_size*diag + tx]);
  }
  maxes[threadIdx.x] = cur_max;

  for (int offset=blockDim.x/2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      maxes[threadIdx.x] = max(maxes[threadIdx.x], maxes[threadIdx.x + offset]);
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) d_block_score[blockIdx.x] = max(d_block_score[blockIdx.x], maxes[0]);

  /*
  __syncthreads();

	if (diag == n_diags-1 && blockIdx.x == 0 && threadIdx.x == 0) {
		for (int y=0; y<n_diags; ++y) {
			for (int x=0; x<diag_size; ++x) {
					printf("%d ", d_score[diag_size*y+x]);
			}
			printf("\n");
    }
    printf("\n");
  }
  */

}

int smith_waterman(int blocks_per_grid, int threads_per_block, char *a, char *b, int a_len, int b_len) {
  
  int *h_score, *d_score; // score matrix in (diag, idx) coordinates
  char *d_a, *d_b;
  int n_score = (a_len + b_len + 1) * (min(a_len, b_len) + 1); // TODO: allocaton too large, use three rows to store instead of n.
  size_t size_score = n_score * sizeof(int);

	h_score = (int *) malloc(size_score);
	cudaMalloc(&d_score, size_score);
	cudaMalloc(&d_a, a_len * sizeof(int));
  cudaMalloc(&d_b, b_len * sizeof(int));

	cudaMemcpy(d_a, a, a_len, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, b_len, cudaMemcpyHostToDevice);

  size_t size_block_score = blocks_per_grid * sizeof(int);
  int *h_block_score, *d_block_score;
  h_block_score = (int *) malloc(size_block_score);
  cudaMalloc(&d_block_score, size_block_score);

  for (int diag=2; diag < a_len + b_len + 1; ++diag) {
	  CalcScoreKernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(int)>>>(d_score, d_a, d_b, a_len, b_len, diag, d_block_score);
  }

  cudaMemcpy(h_block_score, d_block_score, size_block_score, cudaMemcpyDeviceToHost);

  int max = *(std::max_element(h_block_score, h_block_score + blocks_per_grid));

  free(h_score);
  free(h_block_score);
	cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_score);
  cudaFree(d_block_score);

	return max;
}