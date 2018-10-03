Based on https://github.com/arbor-sim/arbor.git

> module load googletest/1.8.1-CrayGNU-18.08

> make clean; make

```
g++ -Igoogletest/1.8.1-CrayGNU-18.08/include/gtest -I./include -I./arbor -std=c++14 -g -c test_algorithms.cpp

g++ -Igoogletest/1.8.1-CrayGNU-18.08/include/gtest -I./include -I./arbor -std=c++14 -g -c test.cpp

g++ -Igoogletest/1.8.1-CrayGNU-18.08/include/gtest -I./include -I./arbor \
-std=c++14 -g test_algorithms.o test.o -Lgoogletest/1.8.1-CrayGNU-18.08/lib64 \
-lgtest -lpthread -o GNU_6.2.0.exe 
```

> ./GNU_6.2.0.exe 

```
[==========] Running 1 test from 1 test case.
[----------] Global test environment set-up.
[----------] 1 test from algorithms
[ RUN      ] algorithms.sum
[       OK ] algorithms.sum (0 ms)
[----------] 1 test from algorithms (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test case ran. (1 ms total)
[  PASSED  ] 1 test.
```
