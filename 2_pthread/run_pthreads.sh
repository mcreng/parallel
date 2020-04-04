g++ -std=c++11 -lpthread main.cpp pthreads_smith_waterman_skeleton.cpp -o pthreads_smith_waterman
num=2
test=./datasets/1k.in
echo $num
echo $test
./pthreads_smith_waterman $test $num
