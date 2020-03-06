/*
 * Do not change this file.
 * This is a mpi version of the smith waterman algorithm
 * Compile: mpic++ -std=c++11 main.cpp mpi_smith_waterman_skeleton.cpp -o mpi_smith_waterman
 * Run: mpiexec -n <number of process> ./mpi_smith_waterman <input file>
 */


#pragma once

#include "mpi.h"


using namespace std;

const int MATCH = 3, MIS = -3, GAP = 2;

int smith_waterman(int my_rank, int p, MPI_Comm comm, char *a, char *b, int a_len, int b_len);

// return score of substitution matrix
inline int sub_mat(char x, char y) {
	return x == y ? MATCH : MIS;
}
