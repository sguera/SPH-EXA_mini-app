all: main

main:
	mpic++ -g -O2 -Wall -Wextra -std=c++14 main.cpp -fopenmp

N=4
run: main
	mpirun --hostfile hosts -n $(N) ./a.out