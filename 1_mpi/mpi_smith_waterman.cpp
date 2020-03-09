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
    std::vector<int> v_start_idx = std::vector<int>(p, 0);
    std::vector<int> v_end_idx = std::vector<int>(p, 0);
    std::vector<int> v_len_idx = std::vector<int>(p, 0);
    std::vector<int> v_cum_idx = std::vector<int>(p, 0);  // for gatherv
    int start_idx, end_idx;

    for (int iter = 1; iter <= a_len + b_len - 1; ++iter) {
        MPI_Bcast(&diagonal_t_1[0], max_diag_size, MPI_INTEGER, 0, comm);

        // len stores how many values are there in the current diagonal.
        int len = 0;
        if (iter < std::min(a_len, b_len) - 1)
            len = iter + 2;
        else if (iter < std::max(a_len, b_len) - 1)
            len = std::min(a_len, b_len) + 1;
        else
            len = std::min(a_len, b_len) - (iter - std::max(a_len, b_len));

        if (my_rank == 0) {
            int ceil = std::ceil(len / (float)p);
            int floor = std::floor(len / (float)p);
            int mod = len % p;
            if (mod == 0) mod = p;

            for (int rank = 0; rank < mod; ++rank) {
                v_start_idx.at(rank) = rank * ceil;
                v_end_idx.at(rank) = v_start_idx.at(rank) + ceil;
                v_len_idx.at(rank) = ceil;
            }
            for (int rank = mod; rank < p; ++rank) {
                v_start_idx.at(rank) = mod * ceil + (rank - mod) * floor;
                v_end_idx.at(rank) = v_start_idx.at(rank) + floor;
                v_len_idx.at(rank) = floor;
            }

            for (int cum = 0, rank = 0; rank < p; ++rank) {
                v_cum_idx.at(rank) = cum;
                cum += v_len_idx.at(rank);
            }
        }

        MPI_Scatter(&v_start_idx[0], 1, MPI_INTEGER, &start_idx, 1, MPI_INTEGER, 0, comm);
        MPI_Scatter(&v_end_idx[0], 1, MPI_INTEGER, &end_idx, 1, MPI_INTEGER, 0, comm);

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
        diagonal_t = std::vector<int>(len, 0);

        MPI_Reduce(&diagonal_t_p[0], &diagonal_t[0], len, MPI_INTEGER, MPI_SUM, 0, comm);

        // if (my_rank == 0) {
        //     std::cout << "iteration " << iter << ": ";
        //     for (int diag : diagonal_t) std::cout << diag << " ";
        //     std::cout << std::endl;
        // }
        diagonal_t_2 = diagonal_t_1;
        if (my_rank == 0) {
            curr_max = std::max(curr_max, *std::max_element(diagonal_t.begin(), diagonal_t.end()));
            diagonal_t_1 = diagonal_t;
        }
    }

    return curr_max;
}