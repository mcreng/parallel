/**
 * Name: TSE Ho Nam
 * Student id: hntse
 * ITSC email: hntse@connect.ust.hk
*/

#include "pthreads_smith_waterman.h"

#include <pthread.h>
#include <semaphore.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/* 2D array storing the scores */
int **score;
/* array storing max of each thread */
int *max_vals;
/* barrier */
pthread_barrier_t barrier;

void handler(int p, long rank, char *a, char *b, int a_len, int b_len) {
    /* Range is [block_l, block_r) */
    int block_l = rank * b_len / p;
    int block_r = block_l + b_len / p;
    int n_diagonal = a_len + p - 1;
    int max_score = 0;
    for (int i = 0; i < n_diagonal; i++) {
        int row = i - rank + 1;
        if (row > 0 && row <= a_len) {
            for (int j = 1 + block_l; j <= block_r; j++) {
                score[row][j] = max(0,
                                    max(score[row - 1][j - 1] + sub_mat(a[row - 1], b[j - 1]),
                                        max(score[row - 1][j] - GAP,
                                            score[row][j - 1] - GAP)));
                max_score = max(max_score, score[row][j]);
            }
        }
        /* Barrier */
        pthread_barrier_wait(&barrier);
    }
    max_vals[rank] = max_score;
}

int smith_waterman(int p, char *a, char *b, int a_len, int b_len) {
    score = (int **)malloc(sizeof(int *) * (a_len + 1));
    for (int i = 0; i <= a_len; i++) {
        score[i] = (int *)calloc(b_len + 1, sizeof(int));
    }
    max_vals = (int *)malloc(sizeof(int) * p);
    std::vector<std::thread *> threads(p, nullptr);
    pthread_barrier_init(&barrier, nullptr, p);
    for (long rank = 0; rank < p; rank++) {
        threads[rank] = new std::thread(handler, p, rank, a, b, a_len, b_len);
    }

    for (long rank = 0; rank < p; rank++) {
        threads[rank]->join();
        delete threads[rank];
        threads[rank] = nullptr;
    }

    // for (int i = 0; i < a_len + 1; i++) {
    //     for (int j = 0; j < b_len + 1; j++) {
    //         cout << score[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    int max = *(std::max_element(max_vals, max_vals + p));

    for (int i = 0; i <= a_len; i++) {
        free(score[i]);
    }
    free(score);
    free(max_vals);
    return max;
}