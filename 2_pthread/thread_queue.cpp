#include <pthread.h>
#include <semaphore.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <class T>
class ThreadQueue {
   private:
    std::queue<T> _queue;  // queue storing data
    T _default;            // data to push into queue when stop
    int _p;                // number of threads
    std::mutex _mutex;     // mutex controlling access to queue
    sem_t _sem;            // semaphore storing length of queue

   public:
    ThreadQueue(int p, T d) : _default{d}, _p{p} {
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

    void destroy() {
        for (int i = 0; i < _p; ++i) {
            enqueue(_default);
        }
        std::cout << "Destroyed" << std::endl;
    }
};

bool has_stopped;
ThreadQueue<std::pair<int, int>> *queue;

void handler(long rank) {
    while (true) {
        if (has_stopped) break;
        std::pair<int, int> elem = queue->dequeue();
        if (elem.first != -1 && elem.second != -1)
            printf("Thread %ld got (%d, %d)\n", rank, elem.first, elem.second);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main(int argc, char **argv) {
    assert(argc == 2 && "Insufficient number of parameters.");
    int p = atoi(argv[1]);

    queue = new ThreadQueue<std::pair<int, int>>(p, std::pair<int, int>(-1, -1));

    std::vector<std::thread *> threads(p, nullptr);
    has_stopped = false;

    for (long rank = 0; rank < p; rank++) {
        threads[rank] = new std::thread(handler, rank);
    }

    for (int i = 0; i < 1000; i++) {
        queue->enqueue(std::pair<int, int>(i, 2 * i));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    has_stopped = true;
    queue->destroy();

    for (long rank = 0; rank < p; rank++) {
        threads[rank]->join();
        delete threads[rank];
        threads[rank] = nullptr;
    }

    delete queue;
}