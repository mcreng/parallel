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
#include <vector>

void scheduler_v1(int my_rank, int p, int len, int &start_idx, int &end_idx, std::vector<int> &v_len, std::vector<int> &v_disp, int &p_null) {
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

    if (ceil == 1)
        p_null = mod;
    else
        p_null = p;

    // if (my_rank == 0) {
    std::fill(v_len.begin(), v_len.begin() + mod, ceil);
    std::fill(v_len.begin() + mod, v_len.end(), ceil - 1);
    for (int cumsum = 0, i = 0; i < p; i++) {
        v_disp[i] = cumsum;
        cumsum += v_len[i];
    }
    // }
}

void scheduler_v2(int my_rank, int p, int len, int &start_idx, int &end_idx, std::vector<int> &v_len, std::vector<int> &v_disp, int &p_null) {
    const int BUFFER_SIZE = 32;

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

        p_null = _p + 1;

        // if (my_rank == 0) {
        std::fill(v_len.begin(), v_len.begin() + _p, BUFFER_SIZE);
        v_len[_p] = len - BUFFER_SIZE * _p;
        std::fill(v_len.begin() + _p + 1, v_len.end(), 0);
        for (int cumsum = 0, i = 0; i < p; i++) {
            v_disp[i] = cumsum;
            cumsum += v_len[i];
        }
        // }

    } else
        scheduler_v1(my_rank, p, len, start_idx, end_idx, v_len, v_disp, p_null);
}

int smith_waterman(int my_rank, int p, MPI_Comm comm, char *a, char *b, int a_len, int b_len) {
    int curr_max = 0;
    MPI_Bcast(&a_len, 1, MPI_INT, 0, comm);
    MPI_Bcast(&b_len, 1, MPI_INT, 0, comm);
    int min_len = std::min(a_len, b_len);
    int max_len = std::max(a_len, b_len);

    if (my_rank != 0) {
        a = new char[a_len + 1];
        b = new char[b_len + 1];
    }
    MPI_Bcast(a, a_len + 1, MPI_CHAR, 0, comm);
    MPI_Bcast(b, b_len + 1, MPI_CHAR, 0, comm);

    int max_diag_size = min_len + 2;

    std::vector<int> diagonal_t_p = std::vector<int>(max_diag_size, 0);  // current diagonal (partial)
    std::vector<int> diagonal_t = std::vector<int>(max_diag_size, 0);
    std::vector<int> diagonal_t_1 = std::vector<int>(max_diag_size, 0);  // diagonal before current
    std::vector<int> diagonal_t_2 = std::vector<int>(max_diag_size, 0);  // diagonal before diagonal_t_1

    std::vector<int> v_len = std::vector<int>(p, 0);
    std::vector<int> v_disp = std::vector<int>(p, 0);

    int start_idx, end_idx;
    int prev_len = 0;
    int p_null = p;  // first prcoess id that does not need to do any work

    for (int iter = 1; iter <= a_len + b_len - 1; ++iter) {
        // len stores how many values are there in the current diagonal.
        int len = 0;
        if (iter < min_len - 1)
            len = iter + 2;
        else if (iter < max_len - 1)
            len = min_len + 1;
        else
            len = a_len + b_len - iter;

        scheduler_v2(my_rank, p, len, start_idx, end_idx, v_len, v_disp, p_null);

        if (prev_len > 0 && p_null > 0) {
            if (my_rank == 0) {
                for (int rank = 1; rank < p_null; ++rank)
                    MPI_Send(&diagonal_t_1[0], len + 1, MPI_INT, rank, iter, comm);
            } else if (my_rank < p_null) {
                MPI_Recv(&diagonal_t_1[0], len + 1, MPI_INT, 0, iter, comm, MPI_STATUS_IGNORE);
            }
        }
        // MPI_Bcast(&diagonal_t_1[0], len + 1, MPI_INT, 0, comm);

        if (end_idx > start_idx) {
            // std::fill(diagonal_t_p.begin(), diagonal_t_p.begin() + len, 0);
            for (int j = start_idx; j < end_idx; ++j) {
                int x = std::min(iter, a_len - 1) + 1 - j;
                int y = iter + 1 - x;

                if (x == 0 || y == 0) {
                    diagonal_t_p[j] = 0;
                } else {
                    if (iter < a_len) {
                        diagonal_t_p[j] = std::max(
                            std::max(diagonal_t_1[j - 1] - GAP, diagonal_t_1[j] - GAP),
                            diagonal_t_2[j - 1] + sub_mat(a[x - 1], b[y - 1]));
                    } else if (iter == a_len) {
                        diagonal_t_p[j] = std::max(
                            std::max(diagonal_t_1[j] - GAP, diagonal_t_1[j + 1] - GAP),
                            diagonal_t_2[j] + sub_mat(a[x - 1], b[y - 1]));
                    } else {
                        diagonal_t_p[j] = std::max(
                            std::max(diagonal_t_1[j] - GAP, diagonal_t_1[j + 1] - GAP),
                            diagonal_t_2[j + 1] + sub_mat(a[x - 1], b[y - 1]));
                    }
                }
            }
        }
        // std::fill(diagonal_t_2.begin(), diagonal_t_2.begin() + len, 0);

        MPI_Gatherv(&diagonal_t_p[start_idx], end_idx - start_idx, MPI_INT, &diagonal_t_2[0], &v_len[0], &v_disp[0], MPI_INT, 0, comm);
        // MPI_Allgatherv(&diagonal_t_p[start_idx], end_idx - start_idx, MPI_INT, &diagonal_t_2[0], &v_len[0], &v_disp[0], MPI_INT, comm);

        curr_max = std::max(curr_max, *std::max_element(diagonal_t_p.begin() + start_idx, diagonal_t_p.begin() + end_idx));

        prev_len = len;
        std::swap(diagonal_t_2, diagonal_t_1);
    }

    if (my_rank != 0) {
        delete[] a;
        delete[] b;
    }

    if (my_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &curr_max, 1, MPI_INT, MPI_MAX, 0, comm);
    } else {
        MPI_Reduce(&curr_max, &curr_max, 1, MPI_INT, MPI_MAX, 0, comm);
    }
    return curr_max;
}