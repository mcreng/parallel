mpic++ -std=c++11 main.cpp mpi_smith_waterman_skeleton.cpp -o mpi_smith_waterman
num=2
test=./datasets/1k.in
echo $num
echo $test
mpiexec -n $num --hostfile ~/hostfile ./mpi_smith_waterman $test
