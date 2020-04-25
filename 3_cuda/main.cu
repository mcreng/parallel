/*
 * Do not change this file
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <string>
#include <chrono>

#include "cuda_smith_waterman.h"

using namespace std;
using namespace std::chrono;

namespace utils {
	int a_len;
	int b_len;
	char *a;
	char *b;

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if(inputf){
			inputf >> a_len;
			inputf >> b_len;

			//assert((a_len + b_len) < (1024 * 20));

			a = (char *)malloc(sizeof(char) * (a_len + 1));
			b = (char *)malloc(sizeof(char) * (b_len + 1));

			inputf.get();

			inputf.getline(a, a_len + 1);
			inputf.getline(b, b_len + 1);
		}
		inputf.close();
		return 0;
	}
}

int main(int argc, char **argv) {
	assert(argc > 1 && "Input file was not found!");
	string filename = argv[1];
	int num_blocks_per_grid = atoi(argv[2]);
	int num_threads_per_block = atoi(argv[3]);

	assert(utils::read_file(filename) == 0);

	char *a, *b;

	a = (char *)malloc(sizeof(char) * (utils::a_len + 1));
	b = (char *)malloc(sizeof(char) * (utils::b_len + 1));

	memcpy(a, utils::a, (utils::a_len + 1) * sizeof(char));
	memcpy(b, utils::b, (utils::b_len + 1) * sizeof(char));

#ifdef DEBUG
		cout << a << endl;
		cout << utils::a << endl;

		cout << b << endl;
		cout << utils::b << endl;
#endif

	cudaDeviceReset();
	auto t_start = chrono::high_resolution_clock::now();

	cudaEvent_t cuda_start, cuda_end;
	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_end);
	float kernel_time;

	cudaEventRecord(cuda_start);
	int aln_score = smith_waterman(num_blocks_per_grid, num_threads_per_block, a, b, utils::a_len, utils::b_len);
	cudaEventRecord(cuda_end);

	cudaEventSynchronize(cuda_start);
	cudaEventSynchronize(cuda_end);
	cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);

	GPUErrChk(cudaDeviceSynchronize());

	auto t_end = chrono::high_resolution_clock::now();

	cout << "Max score: "<< aln_score << endl;
	fprintf(stderr, "Elapsed Time: %.9lf s\n",
			duration_cast<nanoseconds>(t_end - t_start).count() / pow(10, 9));
	fprintf(stderr, "Driver Time: %.9lf s\n", kernel_time / pow(10, 3));

	free(a);
	free(b);
	free(utils::a);
	free(utils::b);

	return 0;
}
