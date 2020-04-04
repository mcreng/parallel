/*
 * Do not change this file.
 * This is a pthread version of the smith waterman algorithm
 * Compile: g++ -std=c++11 -lpthread main.cpp pthreads_smith_waterman_skeleton.cpp -o pthreads_smith_waterman
 * Run: ./pthread_smith_waterman <input file> <number of threads>
 */


#pragma once


using namespace std;

const int MATCH = 3, MIS = -3, GAP = 2;

int smith_waterman(int num_threads, char *a, char *b, int a_len, int b_len);

// return score of substitution matrix
inline int sub_mat(char x, char y) {
	return x == y ? MATCH : MIS;
}

namespace utils {
	int idx(int i, int j, int N);
}
