#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'
BOLD='\e[1m'

EXPECTED_OUTPUT_FILE=expected_constants_n25_s0.txt
OUTPUT_FILE=constants.txt

BIN_PATH=../bin/
OMP_BIN=$BIN_PATH/omp.app
MPIOMP_BIN=$BIN_PATH/mpi+omp.app
BIN_PARAMS="-n 25 -s 0"

EXIT_CODE_PASSED=1
EXIT_CODE_FAILED=0
EXIT_CODE_BINARY_MISSING=2

run_test() {
    name=$1
    binary=$2
    printf "${NOCOLOR}\n"

    if test -f $binary; then
        $binary $BIN_PARAMS

        cmp --silent $EXPECTED_OUTPUT_FILE $OUTPUT_FILE && printf "${GREEN}${BOLD}$name Correctness test PASSED\n" ||
                { printf "${RED}${BOLD}Correctness test FAILED\nOutput file diff:\n"; diff $EXPECTED_OUTPUT_FILE $OUTPUT_FILE; return $EXIT_CODE_FAILED; }
        return $EXIT_CODE_PASSED
    else
        printf "${RED}${BOLD}$name binary does not exist in Path: $BIN_PATH, skipping test\n"
        return $EXIT_CODE_BINARY_MISSING
    fi
}

print_verdict() {
    result_code=$1

    if [ $result_code -eq $EXIT_CODE_PASSED ]; then printf "${GREEN}${BOLD}PASSED\n";
    elif [ $result_code -eq $EXIT_CODE_FAILED ]; then printf "${RED}${BOLD}FAILED\n";
    else printf "${RED}${BOLD}BINARY MISSING\n";
    fi
}

run_test "OpenMP" $OMP_BIN; omp_result=$?
run_test "MPI+OpenMP" $MPIOMP_BIN; mpiomp_result=$?

printf "\n${NOCOLOR}CORRECTNESS TEST SUMMARY\n"
printf "${NOCOLOR}OpenMP "; print_verdict $omp_result 
printf "${NOCOLOR}MPI+OpenMP "; print_verdict $mpiomp_result
printf "\n"
