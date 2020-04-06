g++ -std=c++11 main.cpp pthreads_smith_waterman.cpp -o pthreads_smith_waterman -lpthread
num=4
test=./datasets/1k.in
echo $num
echo $test
./pthreads_smith_waterman $test $num
