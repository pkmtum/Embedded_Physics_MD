
from __future__ import print_function
import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt


def plot_single_norm(norm_vals, name, max_steps=-1):
  f, ax = plt.subplots(1)
  ax.semilogy(norm_vals[:max_steps])
  ax.set_ylabel('Total {} norm'.format(name))
  ax.set_xlabel('Iteration')
  ax.grid(ls='dashed')
  plt.savefig('plot_grad_norm_{}.pdf'.format(name), tight_layout=True)
  plt.close(f)

step_max = -1
layers = 3

g = np.loadtxt('grad_norm.txt')
g_names = np.loadtxt('grad_norm_names.txt', delimiter=' ', dtype='str')


if layers==4:
  col_idx = [0,0,1,1,2,2,3,3,4,4,5,5,0,0,1,1,2,2,3,3,4,4,5,5,0]
elif layers==3:
  col_idx = [0,0,1,1,2,2,3,3,4,4,0,0,1,1,2,2,3,3,4,4,0]
else:
  raise Exception('Color coding not implemented.')

enc_vars = (layers + 2)*2

enc_tot_norm = (g[:, 0:enc_vars] ** 2).sum(axis=1) ** (0.5)
dec_tot_norm = (g[:, enc_vars:-1] ** 2).sum(axis=1) ** (0.5)
tot_norm = (g[:, :-1] ** 2).sum(axis=1) ** (0.5)

try:
  cols = plt.rcParams['axes.color_cycle']
except:
  print('use new version.')
try:
  cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
except:
  print('use old version.')
  
plot_single_norm(dec_tot_norm, 'decoder', step_max)
plot_single_norm(enc_tot_norm, 'encoder', step_max)
plot_single_norm(tot_norm, '', step_max)

f, ax = plt.subplots(3, sharex=True)

for idx, name in enumerate(g_names):
  if 'enc' in name and 'weight' in name:
    ax[0].semilogy(g[:step_max, idx], ls='-', c=cols[col_idx[idx]])
  elif 'enc' in name and 'bias' in name:
    ax[0].semilogy(g[:step_max, idx], ls='--', c=cols[col_idx[idx]])
  elif 'dec' in name and 'weight' in name:
    ax[1].semilogy(g[:step_max, idx], ls='-', c=cols[col_idx[idx]])
  elif 'dec' in name and 'bias' in name:
    ax[1].semilogy(g[:step_max, idx], ls='--', c=cols[col_idx[idx]])
  elif 'total_norm' in name:
    ax[2].semilogy(tot_norm[:step_max], c=cols[col_idx[0]])

ax[0].set_ylabel('Encoder norm')    
ax[1].set_ylabel('Decoder norm')
ax[2].set_ylabel('Totoal norm')

for a in ax:
  a.grid(ls='dashed')

#plt.show()
plt.savefig('plot_grad_norm.pdf', tight_layout=True)
plt.close(f)


