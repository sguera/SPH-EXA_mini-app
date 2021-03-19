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

print(d[:,0].shape)

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
fx = d[:,14]
fy = d[:,15]
fz = d[:,16]

x = np.array(x)
y = np.array(y)
z = np.array(z)
fx = np.array(fx)
fy = np.array(fy)
fz = np.array(fz)

radius = np.sqrt(x*x + y*y + z*z)
gravity = (fx*x + fy*y + fz*z) / radius

to_plot = gravity

mask = abs(z / h) < 1.0

cm = plt.cm.get_cmap('RdYlBu')

plt.style.use('dark_background')

# Create figure
plt.figure(figsize=(10,8))

# Plot 2D projection a middle cut
sc = plt.scatter(radius, gravity, s=1.0, label="Sedov", vmin=min(to_plot), vmax=max(to_plot))
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

