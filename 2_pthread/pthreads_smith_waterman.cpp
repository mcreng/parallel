/**
 * Name: TSE Ho Nam
 * Student id: hntse
 * ITSC email: hntse@connect.ust.hk
*/

#include "pthreads_smith_waterman.h"

#include <pthread.h>
#include <semaphore.h>

#include <iostream>
#include <queue>
#include <vector>

template <class T>
class ThreadQueue {
   private:
    std::queue<T> _queue;    // queue storing data
    T _default;              // data to push into queue when stop
    int _p;                  // number of threads
    pthread_mutex_t _mutex;  // mutex controlling access to queue
    sem_t _sem;              // semaphore storing length of queue

   public:
    ThreadQueue(int p, T d) : _default{d}, _p{p} {
        pthread_mutex_init(&_mutex, nullptr);
        sem_init(&_sem, 0, 0);
    };

    void enqueue(T data) {
        pthread_mutex_lock(&_mutex);
        sem_post(&_sem);
        _queue.push(data);
        pthread_mutex_unlock(&_mutex);
    };

    T dequeue() {
        sem_wait(&_sem);
        pthread_mutex_lock(&_mutex);
        T elem = _queue.front();
        _queue.pop();
        pthread_mutex_unlock(&_mutex);
        return elem;
    }

    void destroy() {
        for (int i = 0; i < _p; ++i) {
            enqueue(_default);
        }
        std::cout << "Destroyed" << std::endl;
    }
};

/* 2D array storing the scores */
std::vector<std::vector<int>> *scores;
bool has_stopped;
ThreadQueue<std::pair<int, int>> *q;
char *a, *b;
int a_len, b_len;

void *handler(void *in_rank) {
    long rank = (long)in_rank;

    while (true) {
        if (has_stopped) break;
        std::pair<int, int> elem = q->dequeue();
        if (elem.first != -1 && elem.second != -1)
            printf("Thread %ld got (%d, %d)\n", rank, elem.first, elem.second);
    }
    return nullptr;
}

int smith_waterman(int p, char *a, char *b, int a_len, int b_len) {
    scores = new std::vector<std::vector<int>>(a_len, std::vector<int>(b_len, 0));
    q = new ThreadQueue<std::pair<int, int>>(p, std::pair<int, int>(-1, -1));
    std::vector<pthread_t> threads(p);
    ::a = a;
    ::b = b;
    ::a_len = a_len;
    ::b_len = b_len;

    for (long rank = 0; rank < p; rank++) {
        pthread_create(&threads[rank], nullptr, handler, (void *)rank);
    }

    q->enqueue({0, 0});

    for (long rank = 0; rank < p; rank++) {
        pthread_join(threads[rank], nullptr);
    }
    delete scores;
    delete q;
    return 0;
}