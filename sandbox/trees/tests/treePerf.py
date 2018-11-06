import os
import math
import numpy as np
import matplotlib as mpl
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import sys

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

scol = ['orange','cyan', 'magenta','brown','orange','green','black','darkviolet']
smark = ('x', 'o', '+', 's', '*', 'd', '^', 'p', '*', 'h', 'H', 'D', '8')

BroadTreeEvrard = np.loadtxt("treePerfBroadTreeEvrard.res")
KdTreeEvrard = np.loadtxt("treePerfKdTreeEvrard.res")
OctreeEvrard = np.loadtxt("treePerfOctreeEvrard.res")
NNFTreeEvrard = np.loadtxt("treePerfNNFTreeEvrard.res")

w = 0.2
plt.figure(figsize=(5,4))

for i in range(0,3):
	plt.bar(i+1-1.5*w, NNFTreeEvrard[i], width=w, color='C0', label=r"nanoflann" if i == 1 else '')
	plt.bar(i+1-w/2, KdTreeEvrard[i], width=w, color='C2', label=r"KdTree" if i == 1 else '')
	plt.bar(i+1+w/2, OctreeEvrard[i], width=w, color='C1', label=r"Octree" if i == 1 else '')
	plt.bar(i+1+1.5*w, BroadTreeEvrard[i], width=w, color='C3', label=r"BroadTree" if i == 1 else '')

plt.xticks([1, 2, 3], ["BuildTree", "FindNeighbors", "Clean"])
plt.xlim(0.5, 3.5)

plt.legend(loc='upper right')

plt.xlabel(r'Evrard (1Mp and 100 neighbors)')
plt.ylabel('Avgerage time (s)')

plt.tight_layout()
plt.savefig(os.path.normpath('treePerfEvrard.pdf'))
plt.draw()


BroadTreeEvrard = np.loadtxt("treePerfBroadTreeSqpatch.res")
KdTreeEvrard = np.loadtxt("treePerfKdTreeSqpatch.res")
OctreeEvrard = np.loadtxt("treePerfOctreeSqpatch.res")
NNFTreeEvrard = np.loadtxt("treePerfNNFTreeSqpatch.res")

plt.figure(figsize=(5,4))

for i in range(0,3):
	plt.bar(i+1-1.5*w, NNFTreeEvrard[i], width=w, color='C0', label=r"nanoflann" if i == 1 else '')
	plt.bar(i+1-w/2, KdTreeEvrard[i], width=w, color='C2', label=r"KdTree" if i == 1 else '')
	plt.bar(i+1+w/2, OctreeEvrard[i], width=w, color='C1', label=r"Octree" if i == 1 else '')
	plt.bar(i+1+1.5*w, BroadTreeEvrard[i], width=w, color='C3', label=r"BroadTree" if i == 1 else '')

plt.xticks([1, 2, 3], ["BuildTree", "FindNeighbors", "Clean"])
plt.xlim(0.5, 3.5)

plt.legend(loc='upper right')

plt.xlabel(r'Square Patch (10Mp and 450 neighbors)')
plt.ylabel('Avgerage time (s)')

plt.tight_layout()
plt.savefig(os.path.normpath('treePerfSqpatch.pdf'))
plt.draw()

plt.show()

