/**
 * Name: TSE Ho Nam
 * Student id: 20423612
 * ITSC email: hntse@connect.ust.hk
*/

#include "mpi_smith_waterman.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

void scheduler_v1(int my_rank, int p, int len, int &start_idx, int &end_idx) {
    int mod = len % p;
    int ceil = len / p + (mod != 0);
    if (mod == 0) mod = p;

    if (my_rank < mod) {
        start_idx = my_rank * ceil;
        end_idx = start_idx + ceil;
    } else {
        start_idx = mod * ceil + (my_rank - mod) * (ceil - 1);
        end_idx = start_idx + ceil - 1;
    }
}

void scheduler_v2(int my_rank, int p, int len, int &start_idx, int &end_idx) {
    const int BUFFER_SIZE = 4;

    if (len < BUFFER_SIZE * p) {
        int _p = std::floor(len / (float)BUFFER_SIZE);

        if (my_rank < _p) {
            start_idx = my_rank * BUFFER_SIZE;
            end_idx = start_idx + BUFFER_SIZE;
        } else if (my_rank == _p) {
            start_idx = BUFFER_SIZE * _p;
            end_idx = start_idx + len - BUFFER_SIZE * _p;
        } else {
            start_idx = 0;
            end_idx = 0;
        }

    } else
        scheduler_v1(my_rank, p, len, start_idx, end_idx);
}

int smith_waterman(int my_rank, int p, MPI_Comm comm, char *a, char *b, int a_len, int b_len) {
    int curr_max = 0;
    MPI_Bcast(&a_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&b_len, 1, MPI_INT, 0, comm);

    if (my_rank != 0) {
        a = new char[a_len + 1];
        b = new char[b_len + 1];
    }
    MPI_Bcast(a, a_len + 1, MPI_CHAR, 0, comm);
    MPI_Bcast(b, b_len + 1, MPI_CHAR, 0, comm);

    int max_diag_size = std::min(a_len, b_len) + 2;

    std::vector<int> diagonal_t_p = std::vector<int>(max_diag_size, 0);  // current diagonal (partial)
    std::vector<int> diagonal_t = std::vector<int>(max_diag_size, 0);
    std::vector<int> diagonal_t_1 = std::vector<int>(max_diag_size, 0);  // diagonal before current
    std::vector<int> diagonal_t_2 = std::vector<int>(max_diag_size, 0);  // diagonal before diagonal_t_1
    int start_idx, end_idx;

    for (int iter = 1; iter <= a_len + b_len - 1; ++iter) {
        // len stores how many values are there in the current diagonal.
        int len = 0;
        if (iter < std::min(a_len, b_len) - 1)
            len = iter + 2;
        else if (iter < std::max(a_len, b_len) - 1)
            len = std::min(a_len, b_len) + 1;
        else
            len = std::min(a_len, b_len) - (iter - std::max(a_len, b_len));

        scheduler_v2(my_rank, p, len, start_idx, end_idx);

        if (end_idx > start_idx) {
            diagonal_t_p = std::vector<int>(len, 0);
            for (int j = start_idx; j < end_idx; ++j) {
                int x = std::min(iter, a_len - 1) + 1 - j;
                int y = iter + 1 - x;

                if (x == 0 || y == 0) {
                    diagonal_t_p.at(j) = 0;
                } else {
                    if (iter < a_len) {
                        diagonal_t_p.at(j) = std::max({0, diagonal_t_1.at(j - 1) - GAP, diagonal_t_1.at(j) - GAP, diagonal_t_2.at(j - 1) + sub_mat(a[x - 1], b[y - 1])});
                    } else if (iter == a_len) {
                        diagonal_t_p.at(j) = std::max({0, diagonal_t_1.at(j) - GAP, diagonal_t_1.at(j + 1) - GAP, diagonal_t_2.at(j) + sub_mat(a[x - 1], b[y - 1])});
                    } else {
                        diagonal_t_p.at(j) = std::max({0, diagonal_t_1.at(j) - GAP, diagonal_t_1.at(j + 1) - GAP, diagonal_t_2.at(j + 1) + sub_mat(a[x - 1], b[y - 1])});
                    }
                }
            }
        }
        diagonal_t_2 = std::vector<int>(len, 0);

        MPI_Allreduce(&diagonal_t_p[0], &diagonal_t_2[0], len, MPI_INT, MPI_SUM, comm);

        if (len <= 100) {
            if (my_rank == 0) {
                curr_max = std::max(curr_max, *std::max_element(diagonal_t_2.begin(), diagonal_t_2.end()));
            }
        } else {
            curr_max = std::max(curr_max, *std::max_element(diagonal_t_p.begin(), diagonal_t_p.end()));
            if (my_rank == 0) {
                MPI_Reduce(MPI_IN_PLACE, &curr_max, 1, MPI_INT, MPI_MAX, 0, comm);
            } else {
                MPI_Reduce(&curr_max, &curr_max, 1, MPI_INT, MPI_SUM, 0, comm);
            }
        }

        std::swap(diagonal_t_2, diagonal_t_1);
    }

    return curr_max;
}