/*
 * Do not change this file.
 * This is a serial version of the smith waterman algorithm
 * Compile: g++ -std=c++11 main.cpp serial_smith_waterman.cpp -o serial_smith_waterman
 * Run: ./serial_smith_waterman <input_file>
 */


#pragma once

using namespace std;

const int MATCH = 3, MIS = -3, GAP = 2;

int smith_waterman(char *a, char *b, int a_len, int b_len);

// return score of substitution matrix
inline int sub_mat(char x, char y) {
	return x == y ? MATCH : MIS;
}
