#!/usr/bin/env python3
# coding: utf-8

import json
import sys
import re
import numpy as np
from common import *
import matplotlib
# import matplotlib_terminal
import matplotlib.pyplot as plt

dims = ['xsmall', 'small', 'medium', 'large']
prg_models = ['SphExa_MPI_Check',
              'SphExa_MPI_OpenMP_Cuda_Check',
              'SphExa_MPI_OpenMP_Target_Check']
prg_envs = ['PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-cray', 'PrgEnv-pgi']
cubeside_d = {1: 'xsmall', 6: 'small', 12: 'medium', 672: 'large'}
# -------------------------------
mpi_tasks = [1, 6, 12]
cubeside_dict = {1: 30, 6: 30, 12: 30}
steps_dict = {1: 0, 6: 0, 12: 0}
# -------------------------------
# mpi_tasks = [1, 6, 24]  # jenkins restricted to 1 cnode
# steps_dict = {1: 0, 6: 1, 24: 2, 672: 10}
# cubeside_dict = {1: 30, 6: 30, 24: 100, 672: 300}
# -------------------------------

rows = {}
rows = init_dict(dims, prg_models, prg_envs)

f = open('rpt.json')
d = json.load(f)
f.close()

rows = update_dict(d, rows, cubeside_d)

# {{{ rows_labels
dim_first_key = next(iter(rows))  # should be 'xsmall'
prg_model_first_key = next(iter(rows[dim_first_key]))  # should be 'MPI'
rows_labels = list(rows[dim_first_key][prg_model_first_key].keys())
# -> ['gnu', 'intel', 'cray', 'pgi']
print('rows_labels=', rows_labels)
# }}}

# {{{ columns_labels
columns_labels = list(rows[dim_first_key].keys())
# -> ['MPI', 'MPI_OpenMP_Cuda', 'MPI_OpenMP_Target']
print('columns_labels=', columns_labels)
# }}}

# {{{ shape data for plot:
dim_to_plot = 'xsmall'
for dim_to_plot in dims:
    myplt_data = shape_data(rows, dim_to_plot, dim_first_key,
                            prg_model_first_key)
    print('myplt_data=', myplt_data)
    plot_data(myplt_data, rows_labels, columns_labels, dim_to_plot)
# }}}
