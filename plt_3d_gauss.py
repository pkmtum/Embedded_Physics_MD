
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

w = np.array([10, 1, -10])
cov = np.outer(w,w) + np.diag(np.ones(3))

s = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=1000)

ax = plt.axes(projection='3d')
ax.scatter3D(s[:,0], s[:,1], s[:,2])
plt.show()
