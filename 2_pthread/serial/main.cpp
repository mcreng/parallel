/*
 * Do not change this file
 */


#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>

#include "smith_waterman.h"

using namespace std;

namespace utils{
	int a_len;
	int b_len;

	char *a;
	char *b;

	string a_str, b_str;

	int read_file(string filename) {
		std::ifstream inputf(filename, std::ifstream::in);
		if(inputf){
			inputf >> a_len;
			inputf >> b_len;

			//cleassert((a_len + b_len) < (1024 * 20));

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
	assert(utils::read_file(filename) == 0);

	auto t_start = chrono::high_resolution_clock::now();
	int aln_score = smith_waterman(utils::a, utils::b, utils::a_len, utils::b_len);
	auto t_end = chrono::high_resolution_clock::now();
	chrono::duration<double> diff = t_end - t_start;

	cout << aln_score << endl;
	cerr << "Time: " << diff.count() << " s" << endl;

	return 0;
}
