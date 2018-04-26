# Code for visualizing the 2D points

import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
data = np.loadtxt(filename)

plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(data[:,0], data[:,1], s=3)
plt.show()
