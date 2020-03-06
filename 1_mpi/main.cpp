/*
 * Do not change this file
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <string>
#include <chrono>

#include "mpi_smith_waterman.h"

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

	MPI_Init(&argc, &argv);
	MPI_Comm comm;

	int p; // number of processors
	int my_rank; // my global rank
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);

	char *a, *b;

	if (my_rank == 0) {
		utils::read_file(filename);

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

	}

	MPI_Barrier(comm);
	auto t_start = chrono::high_resolution_clock::now();

	int aln_score = smith_waterman(my_rank, p, comm, a, b, utils::a_len, utils::b_len);

	MPI_Barrier(comm);
	auto t_end = chrono::high_resolution_clock::now();

	if (my_rank == 0) {
		chrono::duration<double> diff = t_end - t_start;

		cout << aln_score << endl;
		cerr << "Time: " << diff.count() << " s" << endl;

		free(a);
		free(b);
		free(utils::a);
		free(utils::b);
	}

	MPI_Finalize();
	return 0;
}
