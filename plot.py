import os
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) < 3:
	print('usage: python plot.py <filename> <n>')
	sys.exit()

file = sys.argv[1]
n = int(sys.argv[2])

d = np.loadtxt(file)

x = d[:,0]
y = d[:,1]
z = d[:,2]
vx = d[:,3]
vy = d[:,4]
vz = d[:,5]
h = d[:,6]
ro = d[:,7]
u = d[:,8]
p = d[:,9]
c = d[:,10]
grad_P_x = d[:,11]
grad_P_y = d[:,12]
grad_P_z = d[:,13]

mask = abs(z / h) < 1.0

cm = plt.cm.get_cmap('RdYlBu')

plt.style.use('dark_background')

# Create figure
plt.figure(figsize=(10,8))

# Plot 2D projection a middle cut
sc = plt.scatter(x[mask], y[mask], c=ro[mask], s=10.0, label="Sedov", vmin=min(ro[mask]), vmax=max(ro[mask]), cmap=cm)
plt.colorbar(sc)

plt.axis('square')
#plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.draw()
plt.title('Density')

# fig = plt.figure()
# ax = Axes3D(fig)

# mask = abs(ro) < 0.25

# ax.scatter(x[mask], y[mask], z[mask], c=ro[mask], s=10.0, label="Sedov", vmin=min(ro[mask]), vmax=max(ro[mask]), cmap=cm)

plt.savefig(file+sys.argv[2]+'.png')

plt.show()
#raw_input('press return to continue')

