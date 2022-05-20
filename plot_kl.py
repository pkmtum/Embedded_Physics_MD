

import numpy as np
import matplotlib.pyplot as plt

interval = 50
loss = np.loadtxt('train_hist_kl.txt')
length = loss.shape[0]
iterations = np.linspace(0, length*interval, length + 1)

f, ax = plt.subplots()
ax.plot(iterations, loss)
ax.set_ylabel(r'$D_\text{KL}(q(\mathbf{x},\mathbf{z})||r_\text{target}(\mathbf{x}) r(\mathbf{z}| \mathbf{x}))$')
ax.set_xlabel(r'Iteration')
ax.grid(ls='dashed')
ax.set_axisbelow(True)
f.savefig('loss.pdf', bbox_inches='tight')