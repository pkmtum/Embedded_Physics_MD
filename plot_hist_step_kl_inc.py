import numpy as np
import matplotlib.pyplot as plt
import sys


def addvertline(axx, xpos):
  axx.axvline(x=xpos, c='C1', ls='--', lw='4')
  axx.text(xpos-1500, 10e14, 'Iteration {}'.format(xpos), rotation=90)

def analysevar(prefix, filenamesave, steps):
  varlist = []
  for step in steps:
    fname = prefix + str(int(step)) + '.txt'
    var = np.mean(np.loadtxt(fname))
    varlist.append(var)
  np.savetxt(filenamesave, np.array(varlist))
  return varlist

#predFile = sys.argv[1]

#data = np.loadtxt(predFile)
#data *= 1.
#np.savetxt(predFile, data)

step = 10

d = np.loadtxt('train_hist.txt')

poi = np.array([])

qvarint = np.array(range(400,66001,400))
try:
  analysevar(prefix='decoded_z_var_', filenamesave='varlist_decoder.txt', steps=qvarint)
  analysevar(prefix='encoded_x_var_', filenamesave='varlist_encoder.txt', steps=qvarint)
except IOError:
  print 'cannot open'
else:
  pass
#quit()
bplotvar = True
try:
  vardecoder = np.loadtxt('varlist_decoder.txt')
  varencoder = np.loadtxt('varlist_encoder.txt')
  bplotvar = True
except IOError:
  print 'cannot open'
  bplotvar = False
else:
  bplotvar = True
  pass

bplotstep = True
rel_step_inc_stored = False
try:
  steps = np.load('stepper_history.npy')
  size_steps = len(steps)
  iteration_step = [s['training_iteration'] for s in steps]
  a_step = [s['a_val'] for s in steps]
  if 'rel_kl_inc' in steps[0]:
    kl_inc_step = [s['rel_kl_inc'] for s in steps]
    rel_step_inc_stored = True
  bplotstep = True
except IOError:
  print 'cannot open stepper file.'
  bplotstep = False
else:
  bplotstep = True
  pass
  

dmin = d.min()
d -= dmin
npitermax = d.shape[0]
npiter = np.arange(1, npitermax+1)

f, ax = plt.subplots(1)
ax.semilogy(npiter, d)
[addvertline(ax, xp) for xp in poi]
ax.set_xlim(left=0)
ax.set_ylabel('Loss')
ax.set_xlabel('Iteration')
ax.grid(ls='dashed')

# 
if bplotstep:
  ax.tick_params(axis='y', labelcolor='C0')
  ax2 = ax.twinx()
  ax2.set_yscale('log')
  ax2.tick_params(axis='y', labelcolor='C1')
  ax2.plot(iteration_step, a_step, 'C1.')
  ax2.set_ylabel(r'Prefactor $a$ of simulation temperature $a\beta$')

plt.savefig('loss_single.pdf', tight_layout=True)
plt.close(f)

if rel_step_inc_stored:
  f, ax = plt.subplots(1)
  ax.semilogy(iteration_step, kl_inc_step)
  ax.set_ylabel('Rel. KL increase')
  ax.set_xlabel('Iteration')
  ax.grid(ls='dashed')
  if bplotstep:
    ax.tick_params(axis='y', labelcolor='C0')
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.plot(iteration_step, a_step, 'C1.')
    ax2.set_ylabel(r'Prefactor $a$ of $a\beta$')
  
  plt.savefig('loss_kl_inc.pdf', tight_layout=True)
  plt.close(f)



if bplotvar:
  f, ax = plt.subplots(1)
  ax.semilogy(qvarint, vardecoder, label=r'Mean $\sigma^2$ of r(z|x)')
  ax.semilogy(qvarint, varencoder, label=r'Mean $\sigma^2$ of q(x|z)', c='C2', ls='--')
  [addvertline(ax, xp) for xp in poi]
  ax.set_ylabel('Variance')
  ax.set_xlabel('Iteration')
  ax.legend(loc=7)  
  ax.grid(ls='dashed')
  plt.savefig('variance.pdf', tight_layout=True)
  plt.close(f)

  f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
  ax[0].semilogy(npiter, d)
  [addvertline(ax[0], xp) for xp in poi]
  ax[0].set_ylabel('Loss')
  ax[0].set_xlim(left=0)
  ax[1].semilogy(qvarint, vardecoder, label=r'Mean $\sigma^2$ of q(x|z)')
  ax[1].semilogy(qvarint, varencoder, label=r'Mean $\sigma^2$ of r(z|x)', ls='--')
  ax[1].set_ylabel('Variance')
  ax[1].set_xlabel('Iteration')
  ax[1].legend()
  ax[0].grid(ls='dashed')
  ax[1].grid(ls='dashed')
  plt.savefig('loss_single_var.pdf', tight_layout=True)
  plt.close(f)
  
  #if rel_step_inc_stored:
  #  f, ax = plt.subplots(1)
    

#f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
f, ax = plt.subplots(1)
ax.semilogy(npiter[0:-1:step], d[0:-1:step])
ax.set_ylabel('Loss')
ax.set_xlabel('Iteration')
ax.grid(ls='dashed')



plt.savefig('loss_filter.pdf', tight_layout=True)
plt.close(f)


means = [np.mean(d[i*step:(i+1)*step]) for i in range(int(npitermax/step))]

f, ax = plt.subplots()

ax.semilogy(npiter[0:-1:step], means)
ax.set_ylabel('Loss')
ax.set_xlabel('Iteration')
ax.grid(ls='dashed')

plt.savefig('loss.pdf', tight_layout=True)
plt.close(f)
