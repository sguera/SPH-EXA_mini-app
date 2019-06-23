import os
import math
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('medium/constants.txt')
iteration = data[:,0]
totalTime = data[:,1]
currentTimeStep = data[:,2]
totalEnergy = data[:,3]
cineticEnergy = data[:,4]
internalEnergy = data[:,5]
totalNeighbors = data[:,6]

maxit = 16000

plt.figure(figsize=(5,4))
plt.plot(iteration[0:maxit], totalTime[0:maxit], marker='x', linestyle='-', linewidth=1, markersize=4, markevery=250, color='black', label=r"Total Time")#($f_{child} = 2, n_{p} = 64$)")
plt.plot(iteration[0:maxit], currentTimeStep[0:maxit], marker='^', linestyle='--', linewidth=1, markersize=4, markevery=250, color='gray', label=r"Current Time-Step")#($f_{child} = 2, n_{p} = 64$)")
plt.legend(loc='lower left')
plt.yscale('log')
plt.xlim(0,maxit)
plt.draw()


plt.figure(figsize=(5,4))
plt.plot(iteration[0:maxit], totalEnergy[0:maxit], marker='x', linestyle='-', linewidth=1, markersize=4, markevery=250, color='red', label=r"Total Energy")#($f_{child} = 2, n_{p} = 64$)")
plt.plot(iteration[0:maxit], cineticEnergy[0:maxit], marker='^', linestyle='--', linewidth=1, markersize=4, markevery=250, color='green', label=r"Total Cinetic Energy")#($f_{child} = 2, n_{p} = 64$)")
plt.plot(iteration[0:maxit], internalEnergy[0:maxit], marker='v', linestyle='--', linewidth=1, markersize=4, markevery=250, color='blue', label=r"Total Internal Energy")#($f_{child} = 2, n_{p} = 64$)")
plt.legend(loc='center left')
plt.yscale('log')
plt.xlim(0,maxit)
plt.draw()

plt.show()