#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'
BOLD='\e[1m'

EXPECTED_OUTPUT_FILE=$(mktemp /tmp/sph-exa-correctness-test-tmp.XXXXXXXXXX)
exec 3>"$EXPECTED_OUTPUT_FILE"
echo "0 1.1e-06 1.1e-06 2.07823e+10 2.07813e+10 1e+06 1734080 
1 2.31e-06 1.21e-06 2.07823e+10 2.07813e+10 999881 1974924 
2 3.641e-06 1.331e-06 2.07823e+10 2.07813e+10 925979 2029579 
3 5.1051e-06 1.4641e-06 2.07823e+10 2.07815e+10 880886 1986601 
4 6.71561e-06 1.61051e-06 2.07825e+10 2.07817e+10 815386 2008816 
5 8.48717e-06 1.77156e-06 2.07827e+10 2.07819e+10 732995 1990868 
6 1.04359e-05 1.94872e-06 2.07829e+10 2.07823e+10 631292 1998260 
7 1.25795e-05 2.14359e-06 2.07833e+10 2.07828e+10 503425 2029484 
8 1.49374e-05 2.35795e-06 2.07838e+10 2.07834e+10 346323 1967792 
9 1.75312e-05 2.59374e-06 2.07844e+10 2.07843e+10 149915 2023388 
10 2.03843e-05 2.85312e-06 2.07853e+10 2.07854e+10 -99238 1989454 " > $EXPECTED_OUTPUT_FILE

OUTPUT_FILE=constants.txt

EXIT_CODE_PASSED=0
EXIT_CODE_FAILED=1
EXIT_CODE_BINARY_MISSING=2

BIN=$1

if [ $# -ne 1 ]; 
     then echo "Usage: $0 <path to binary>"
     exit -1
fi 

run_test() {
    binary=$1
    bin_params=$2
    printf "${NOCOLOR}\n"
    
    if test -f $binary; then
        $binary $bin_params

        cmp --silent $EXPECTED_OUTPUT_FILE $OUTPUT_FILE && printf "${GREEN}${BOLD}$name Correctness test PASSED\n" ||
                { printf "${RED}${BOLD}Correctness test FAILED\nOutput file diff:\n"; diff $EXPECTED_OUTPUT_FILE $OUTPUT_FILE; return $EXIT_CODE_FAILED; }

        return $EXIT_CODE_PASSED
    else
        printf "${RED}${BOLD} $binary binary does not exist. Skipping test\n"
        return $EXIT_CODE_BINARY_MISSING
    fi
}

if test -f $OUTPUT_FILE; then rm $OUTPUT_FILE; fi

run_test $BIN "-n 20 -s 10 --quiet"; ret_code=$?

printf "\n"
rm "$EXPECTED_OUTPUT_FILE"


if [ $ret_code -eq $EXIT_CODE_FAILED ]; then
    exit 1
else
    exit 0
fi


