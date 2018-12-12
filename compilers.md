# DAINT

* ok = compiles and runs
* ~/sbatch.sh dom 2 bin/dom_GNU_4.9.3.exe 1 1 1 1 1 -Cmc "" "" "" "" $SCRATCH/

## -O3 -std=c++14 -g + openmp

| runner | compiles | runs |
|---|---|---|
| gcc/5.3.0 | ok ||
| gcc/6.2.0 | ok ||
| gcc/7.3.0 | ok ||
| gcc/8     | todo ||
|  |  |  |
|pgi/18.7 (slow) | ok ||
|  |  |  |  |  |
|cce/8.6.1| ok ||
|cce/8.7.6| ok ||
|cce/9.0.0| ok ||
|  |  |  |
|intel/18.0.2.199| ok ||
|intel/19| todo ||
|  |  |  |
|clang+llvm/7.0.0-x86_64-linux-sles12.3| ok ||
