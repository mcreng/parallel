/**
 * Name: TSE Ho Nam
 * Student id: hntse
 * ITSC email: hntse@connect.ust.hk
*/

#include "pthreads_smith_waterman.h"

#include <semaphore.h>

#include <algorithm>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <class T>
class ThreadQueue {
   private:
    std::queue<T> _queue;  // queue storing data
    int _p;                // number of threads
    std::mutex _mutex;     // mutex controlling access to queue
    sem_t _sem;            // semaphore storing length of queue

   public:
    ThreadQueue(int p) : _p{p} {
        sem_init(&_sem, 0, 0);
    };

    void enqueue(T data) {
        _mutex.lock();
        sem_post(&_sem);
        _queue.push(data);
        _mutex.unlock();
    };

    T dequeue() {
        sem_wait(&_sem);
        _mutex.lock();
        T elem = _queue.front();
        _queue.pop();
        _mutex.unlock();
        return elem;
    }
};

/* array storing max of each thread */
int *max_vals;
/* thread safe buffers */
ThreadQueue<int> **queues;

void handler(int p, long rank, char *a, char *b, int a_len, int b_len) {
    /* Range is [block_l, block_r) */

    int block_l = rank * b_len / p;
    int block_r = (rank + 1) * b_len / p;
    int width = block_r - block_l;
    int **local_score = (int **)malloc(sizeof(int *) * (a_len + 1));
    for (int i = 0; i <= a_len; ++i) {
        local_score[i] = (int *)calloc(width + 1, sizeof(int));
    }
    int n_diagonal = a_len + p - 1;
    int max_score = 0;
    for (int i = 0; i < n_diagonal; i++) {
        int row = i - rank + 1;
        if (row > 0 && row <= a_len) {
            for (int j = 1; j <= width; j++) {
                local_score[row][j] = max(0,
                                          max(local_score[row - 1][j - 1] + sub_mat(a[row - 1], b[block_l + j - 1]),
                                              max(local_score[row - 1][j] - GAP,
                                                  local_score[row][j - 1] - GAP)));
                max_score = max(max_score, local_score[row][j]);
            }
            if (rank < p - 1) {
                queues[rank + 1]->enqueue(local_score[row][width]);
            }
        }
        if (rank > 0 && row + 1 > 0 && row + 1 <= a_len) {
            local_score[row + 1][0] = queues[rank]->dequeue();
        }
    }
    max_vals[rank] = max_score;
}

int smith_waterman(int p, char *a, char *b, int a_len, int b_len) {
    max_vals = (int *)malloc(sizeof(int) * p);
    std::vector<std::thread *> threads(p, nullptr);
    queues = (ThreadQueue<int> **)malloc(sizeof(ThreadQueue<int> *) * p);
    for (long rank = 0; rank < p; rank++) {
        queues[rank] = new ThreadQueue<int>(p);
    }
    for (long rank = 0; rank < p; rank++) {
        threads[rank] = new std::thread(handler, p, rank, a, b, a_len, b_len);
    }

    for (long rank = 0; rank < p; rank++) {
        threads[rank]->join();
        delete threads[rank];
        delete queues[rank];
    }

    int max = *(std::max_element(max_vals, max_vals + p));

    free(queues);
    free(max_vals);
    return max;
}