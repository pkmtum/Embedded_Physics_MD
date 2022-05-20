#!/usr/bin/python



## not displying the plots
import matplotlib as mpl
mpl.use('Agg')

## for font size adoption
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 20}
rc('font', **font)
#rc('text', usetex=True)
leg = {'fontsize': 18}#,
          #'legend.handlelength': 2}
rc('legend', **leg)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import os
#import sphviewer as sph

import matplotlib.mlab as mlab





# adopted from http://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
def myplot(x, y, nb=32, xsize=500, ysize=500):   
  xmin = np.min(x)
  xmax = np.max(x)
  ymin = np.min(y)
  ymax = np.max(y)

  x0 = (xmin+xmax)/2.
  y0 = (ymin+ymax)/2.

  pos = np.zeros([3, len(x)])
  pos[0,:] = x
  pos[1,:] = y
  w = np.ones(len(x))

  P = sph.Particles(pos, w, nb=nb)
  S = sph.Scene(P)
  S.update_camera(r='infinity', x=x0, y=y0, z=0,xsize=xsize, ysize=ysize)
  R = sph.Render(S)
  R.set_logscale()
  img = R.get_image()
  extent = R.get_extent()
  for i, j in zip(xrange(4), [x0,x0,y0,y0]):
    extent[i] += j

  print extent
  return img, extent


def UcGK(X, thetaCurr):
  s = 0
  dimCoarse = X.shape[0]
  
  mu = np.zeros( dimCoarse )
  invcov = np.zeros( [ dimCoarse, dimCoarse ] )
  
  for i in range(0, thetaCurr.shape[0]):
    mu = gkMu[i,:]
    for k in range(0, dimCoarse):
      invcov[ k,k ] = 1/gkCov[i,k]
    
    aa = X-mu
    s = s + thetaCurr[i]* np.exp( -0.5* (aa.dot(invcov)).dot(aa) )
    #print -0.5* (aa.dot(invcov)).dot(aa) 
  s = s*beta
  return -s

def Uc(X, p, pol):
  
  nPar = par.shape[0]
  s = 0
  
  for i in range(0, nPar):
    s = s + p[i]*X[0]**pol[i,0]* X[1]**pol[i,1]
  
  return -s

def UcSinCos(X, theta):
  fmin = 0.1
  fmax = 3
  nF = 30
  feq = np.linspace(fmin, fmax,nF)
  
  nPar = theta.shape[0]
  s = 0
  
  count = 0
  for i in range(0,X.shape[0]):
    for j in range(0,feq.shape[0]):
      s = s + theta[count]* np.sin( np.pi * feq[j] * X[i]) + theta[count+nPar/2]* np.cos( np.pi * feq[j] * X[i])
      count = count + 1
  
  return -s
  
def UcSinCosMult(X, dim, setting, theta):
  
  freqMult = setting[:,dim]
  sinMult = setting[:,dim+1]
  
  nPar = setting.shape[0]
  s = 0
  
  count = 0
  
  for p in range(0,nPar/2):
    arg = 1.0
    for i in range(0,X.shape[0]):
      arg = arg * X[i]**setting[p,i]
      
    s = s + theta[p]* np.sin( freqMult[p] * np.pi * arg ) + theta[p+nPar/2]*np.cos( freqMult[p+nPar/2] *np.pi * arg)
  s = s*beta
  
  tempS = 0
  for i in range(0,X.shape[0]):
    tempS = tempS + X[i]*X[i]
    
  temps = tempS*0.5
  
  s = s + tempS
  
  return -s


#x = np.linspace(0,5,500)
#a=1
#b=5

#p = pow(b,a)*pow(x,a-1)*np.exp(-b*x)

# energy surface
#par = np.loadtxt('par_init.txt')
#polyn = np.loadtxt('polynom.txt.txt')







kb = 8.31151601e-3
T = 330.
beta = 1./(kb*T)

print beta

fFormat = '.pdf'





if len(sys.argv) > 1:
  iIterMin = int(sys.argv[1])
  iIterMax = iIterMin + 1
  print 'Start with iteration', iIterMin
else:
  iIterMin = 49
  iIterMax = 50
if len(sys.argv) > 2:
  skipEntry = int(sys.argv[2])


bCluster = True
bAddBasis = False
bBBVI = True
bSamplePost = False

if bCluster:
  os.system('mkdir cg/plots/pot')
  sPath = 'cg'
  sPathPlot = 'cg/plots/pot/'
else:
  sPath = 'dataSim'
  sPathPlot = 'plots/'
  
bSinCosMult = False
sinCosSetting = np.zeros(10)#np.loadtxt('sincos_prop_mat.txt')

bGK = True
gkCov = np.loadtxt(sPath+'/gkCov.txt')
gkMu = np.loadtxt(sPath+'/gkMu.txt')

iDim = 2
Nmax = 526

# indexing array 0: alpha 1: beta1 2: beta2
iCC = 0
iA = 10
iB1 = 321
iB2 = 195
colInd = list()
for i in range(0,iA):
  colInd.append('k')
for i in range(0,iB1):
  colInd.append('b')
for i in range(0,iB2):
  colInd.append('r')
  


M_start = 0
M = 2
M_end = iDim
dimCoarse = iDim

parAll = np.loadtxt(sPath+'/theta.txt')


if bAddBasis:
  iterBasisAdd = np.loadtxt(sPath+'/addBasis_iterOverview.txt', dtype=int)
  # actually evaluate the last iteration before the next basis is added
  iterBasisAdd = iterBasisAdd[1:]
  #iterBasisAdd -= 1
  rangeIter = iterBasisAdd
  
  # color for all added basis
  colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
  colorBasis = [colormap(i) for i in np.linspace(0, 0.99, len(iterBasisAdd))]
  
else:
  rangeIter = np.arange(iIterMin,iIterMax,dtype=int)


countBasis = 0
for ii in np.nditer(rangeIter):
  
  if bAddBasis:
    countBasis = countBasis +1
  else:
    countBasis = gkMu.shape[0]
    
  par = parAll[:countBasis,ii]

  zmin = -40
  zmax = 40
  
  xmin = 0
  xmax = 1
  
  x = np.linspace(xmin, xmax, 200)
  y = np.linspace(xmin, xmax, 200)
  X, Y = np.meshgrid(x, y)
  Z = np.zeros( X.shape )

  if bBBVI:
    iterElboMax =1001

    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
    colorst = [colormap(i) for i in np.linspace(0, 0.9, Nmax)]

    muStore = np.zeros( [dimCoarse, Nmax] )
    sigSqStore = np.zeros( [dimCoarse, Nmax] )
    #elboStore = np.zeros([iterElboMax, Nmax] )

    delta = 0.002
    x = np.arange(0, 1, delta)
    y = np.arange(0, 1, delta)
    X, Y = np.meshgrid(x, y)
    
    for i in range(0,Nmax):
      fileMu = sPath+'/bvi_dp_'+str(i)+'_iter_'+ str(ii)+'_mu.txt'
      fileSigSq = sPath+'/bvi_dp_'+str(i)+'_iter_'+str(ii)+'_sigsq.txt'
      #fileElbo = 'bvi_dp_'+str(i)+'_iter_'+rangeIter[ii]+'_elbo.txt'
      
      mu = np.loadtxt(fileMu)
      sigSq = np.loadtxt(fileSigSq)
      #elbo = np.loadtxt(fileElbo)
      
      muStore[:,i] = mu
      sigSqStore[:,i] = sigSq
      
      #elboStore[:,i] = elbo
    
    
    
    
    
    ############################################################################
    ### Plot the posterior distributions for multidimensional latent space
    ############################################################################
    
    
    f, axarr = plt.subplots(dimCoarse,  dimCoarse, sharex='col', sharey='row')
    f.suptitle(r'Means of $q(\mathbf{X}^{(i)}| \mathbf{x}^{(i)}, \mathbf{\theta})$', fontsize=14)#, ~\dim(\theta)='+str(countBasis)+'$') (r'$\mathbf{X}^i \sim q_i(\mathbf{X}^i| \mathbf{x}^i, \mathbf{\theta}), \dim(\theta)='+str(countBasis)+'$', fontsize = 14)
    
    for i in range(0,dimCoarse):
      for j in range(i+1, dimCoarse):
	icountA = 0
	icountB1 = 0
	icountB2 = 0
	
	for k in range(0,Nmax):
	#CS = plt.contour(X, Y, Z, lw=.5, alpha=0.5)
	#plt.clabel(CS, inline=1, fontsize=10)
	  if colInd[k] == 'k' and icountA ==0 :
	    alphaLegendHandle = axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='v',s=10, linewidths=1, c=colInd[k], alpha=1, label=r'$\alpha$')
	    icountA = 1
	  elif colInd[k] == 'k':
	    axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='v',s=10, linewidths=1, c=colInd[k], alpha=1)
	    
	  if colInd[k] == 'b' and icountB1 == 0:
	    beta1LegendHandle = axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='x',s=10, linewidths=2, c=colInd[k], alpha=1, label=r'$\beta$-1')
	    icountB1 = 1
	  elif colInd[k] == 'b':
	    axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='x',s=10, linewidths=2, c=colInd[k], alpha=1)
	    
	  if colInd[k] == 'r' and icountB2 == 0:
	    beta2LegendHandle = axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='o',s=10, linewidths=1, c=colInd[k], alpha=1, label=r'$\beta$-2')
	    icountB2 = 1
	  elif colInd[k] == 'r':
	    axarr[i,j].scatter(muStore[i,k], muStore[j,k], marker='o',s=10, linewidths=1, c=colInd[k], alpha=1)
	  
	axarr[i,j].set_xlim([0,1])
	axarr[i,j].set_ylim([0,1])
	axarr[i,j].grid()
	if i==0 and j > 0:
	  axarr[i,j].tick_params(labelsize=8)
	  axarr[i,j].xaxis.tick_top()
	  axarr[i,j].set_xlabel(r'$X_' + str(j) +'$', size = 12)
	  axarr[i,j].xaxis.set_label_position("top")
	if j==(dimCoarse-1):
	  axarr[i,j].tick_params(labelsize=8)
	  axarr[i,j].yaxis.tick_right()
	  axarr[i,j].set_ylabel(r'$X_' + str(i) +'$', size = 12)
	  axarr[i,j].yaxis.set_label_position("right")
	  
    for i in range(0,dimCoarse):
      for j in range(0,i+1):
	f.delaxes(axarr[i,j])
    
    f.tight_layout()
    plt.legend(handles=[alphaLegendHandle, beta1LegendHandle, beta2LegendHandle ], scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.14),
      fancybox=False, shadow=False, ncol=3, frameon=True)
    #f.subplots_adjust(top=0.88)
    #f.savefig('scatter_ang_'+str(M_start+1)+'_'+str(M_start+M)+'.pdf')#,bbox_inches='tight')
    #f.title(r'$U_c$ with $\dim(\theta_c)='+str(countBasis)+'$')
    plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ fFormat,bbox_inches='tight', transparent = True)
    plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ '.svg',bbox_inches='tight', transparent = True)
    plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ '.eps',bbox_inches='tight', transparent = True)
    #f.show()
    plt.close()
    
    quit()
    
    ##plt.xlim([0,1])
    ##plt.ylim([0,1])
    #plt.xlabel(r'$X_{'+str(i)+'}$')
    #plt.ylabel(r'$X_{'+str(j)+'}$')
    #plt.title(r'$\mathbf{X}^i \sim q(\mathbf{X}^{(i)}| \mathbf{x}^{(i)}, \mathbf{\theta})$')#, ~\dim(\theta)='+str(countBasis)+'$')
    #plt.legend(scatterpoints=1)
    ##plt.grid()
    ##plt.xlim([-0.05,0.55])
    ##plt.ylim([-0.05,0.5])
    ##plt.legend()
    #plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ fFormat, bbox_inches='tight')
    #plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ '.svg', bbox_inches='tight')
    #plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_mult_dim_basis_' +str(countBasis)+ '.eps', bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    
    
    for i in range(0,dimCoarse):
      for j in range(i+1, dimCoarse):
	
	icountA = 0
	icountB1 = 0
	icountB2 = 0
	
	for k in range(0,Nmax):
	  Z = mlab.bivariate_normal(X, Y, np.sqrt(sigSqStore[i,k]*2), np.sqrt(sigSqStore[j,k]*2), muStore[i,k], muStore[j,k],0)
	  #CS = plt.contour(X, Y, Z, lw=.5, alpha=0.5)
	  #plt.clabel(CS, inline=1, fontsize=10)
	  
	  if colInd[k] == 'k' and icountA ==0 :
	    plt.scatter(muStore[i,k], muStore[j,k], marker='v',s=40, linewidths=1, c=colInd[k], alpha=1, label=r'$\alpha$')
	    icountA = 1
	  elif colInd[k] == 'k':
	    plt.scatter(muStore[i,k], muStore[j,k], marker='v',s=40, linewidths=1, c=colInd[k], alpha=1)
	    
	  if colInd[k] == 'b' and icountB1 == 0:
	    plt.scatter(muStore[i,k], muStore[j,k], marker='x',s=40, linewidths=2, c=colInd[k], alpha=1, label=r'$\beta$-1')
	    icountB1 = 1
	  elif colInd[k] == 'b':
	    plt.scatter(muStore[i,k], muStore[j,k], marker='x',s=40, linewidths=2, c=colInd[k], alpha=1)
	    
	  if colInd[k] == 'r' and icountB2 == 0:
	    plt.scatter(muStore[i,k], muStore[j,k], marker='o',s=40, linewidths=1, c=colInd[k], alpha=1, label=r'$\beta$-2')
	    icountB2 = 1
	  elif colInd[k] == 'r':
	    plt.scatter(muStore[i,k], muStore[j,k], marker='o',s=40, linewidths=1, c=colInd[k], alpha=1)
	    
	  #else:
	  #  plt.scatter(muStore[i,k], muStore[j,k], marker='x',s=40, linewidths=3, c=colInd[k], alpha=1)
	  #plt.annotate(str(k), xy=(muStore[i,k], muStore[j,k]),size=10)
  
	xExt = 0
	#plt.xlim([0,1])
	#plt.ylim([0,1])
	plt.xlabel(r'$X_{'+str(i)+'}$')
	plt.ylabel(r'$X_{'+str(j)+'}$')
	plt.title(r'$\mathbf{X}^i \sim q(\mathbf{X}^{(i)}| \mathbf{x}^{(i)}, \mathbf{\theta})$')#, ~\dim(\theta)='+str(countBasis)+'$')
	plt.legend(scatterpoints=1, transparent = True)
	plt.grid()
	#plt.xlim([-0.05,0.55])
	#plt.ylim([-0.05,0.5])
	#plt.legend()
	plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_dim_' + str(i) + '_' + str(j) + '_basis_' +str(countBasis)+ fFormat, bbox_inches='tight', transparent = True)
	plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_dim_' + str(i) + '_' + str(j) + '_basis_' +str(countBasis)+ '.svg', bbox_inches='tight', transparent = True)
	plt.savefig(sPathPlot+'q_X_wonum_iter_'+str(ii)+'_dim_' + str(i) + '_' + str(j) + '_basis_' +str(countBasis)+ '.eps', bbox_inches='tight', transparent = True)
	#plt.show()
	plt.close()

    #for k in range(0,Nmax):
      
      #plt.plot(np.trim_zeros(elboStore[:,k]), lw=2)
      #plt.ylabel('ELBO')
      #plt.xlabel('iteration')
      #plt.title('N='+str(k))
      #plt.grid()
      #plt.savefig('elbo_iter_149_N_'+str(k)+ fFormat)
      #plt.close()
	
      #for i in range(0,dimCoarse):
	#for j in range(i+1, dimCoarse):
	  
	  #delta = 0.0025
	  #x = np.arange(muStore[i,k]-0.05, muStore[i,k]+0.05, delta)
	  #y = np.arange(muStore[j,k]-0.05, muStore[j,k]+0.05, delta)
	  #X, Y = np.meshgrid(x, y)

	  #plt.scatter(sampleStore[i,:,k], sampleStore[j,:,k], c=colorst[k], alpha=0.4, label=k)
	  
	  #Z = mlab.bivariate_normal(X, Y, np.sqrt(sigSqStore[i,k]), np.sqrt(sigSqStore[j,k]), muStore[i,k], muStore[j,k],0)
	  #CS = plt.contour(X, Y, Z, alpha=0.5)
	  ##plt.clabel(CS, inline=1, fontsize=10)
	  
	  #plt.scatter(muStore[i,k], muStore[j,k], marker='x',s=50, linewidths=3, c=colorst[k], alpha=1, label=k)
	  #plt.annotate(str(k), xy=(muStore[i,k], muStore[j,k]),size=8)
	  
	#xExt = 0
	##plt.xlim([0,1])
	##plt.ylim([0,1])
	#plt.xlabel(r'$X_{'+str(i)+'}$')
	#plt.ylabel(r'$X_{'+str(j)+'}$')
	#plt.title(r'$\mathbf{X}^i \sim q_i(\mathbf{X}^i| \mathbf{x}^i, \mathbf{\theta})$ N=' +str(k))
	#plt.grid()
	##plt.legend()
	#plt.savefig('q_X_iter_149_N_'+str(k)+'_dim_' + str(i) + '_' + str(j) + fFormat)
	##plt.show()
	#plt.close()


  #########################
  ### Plot samples pc   ###
  #########################
  
  filePc = sPath+'/sample_pc_'+str(ii)+'.txt'
  sPc = np.loadtxt(filePc)
   
  heatmap, xedges, yedges = np.histogram2d(sPc[0,:], sPc[1,:], bins=10)
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

  plt.clf()
  plt.imshow(heatmap, extent=extent)
  plt.close()
  #plt.show()
  
  for i in range(0,dimCoarse):
    for j in range(i+1, dimCoarse):
      plt.scatter(sPc[i,:],sPc[j,:], alpha=0.7)
      plt.xlabel(r'$X_{'+str(i)+'}$')
      plt.ylabel(r'$X_{'+str(j)+'}$')
      xExt = 6
      #plt.xlim([0,1])
      #plt.ylim([0,1])
      plt.title(r'$\mathbf{X} \sim p_c(\mathbf{X}|\theta), ~\dim(\theta)='+str(countBasis)+'$')
      plt.grid()
      plt.savefig(sPathPlot+'pc_X_iter_'+str(ii)+'_dim_' + str(i) + '_' + str(j) +'_basis_'+str(countBasis)+ '.png')
      #plt.show()
      plt.close()
      
  for i in range(0,dimCoarse):
    for j in range(i+1, dimCoarse):
      plt.hexbin(sPc[i,:],sPc[j,:], gridsize=40, cmap=cm.jet, bins=None)
      plt.xlabel(r'$X_{'+str(i)+'}$')
      plt.ylabel(r'$X_{'+str(j)+'}$')
      xExt = 6
      #plt.xlim([0,1])
      #plt.ylim([0,1])
      plt.title(r'$\mathbf{X} \sim p_c(\mathbf{X}|\theta), ~\dim(\theta)='+str(countBasis)+'$')
      plt.grid()
      plt.savefig(sPathPlot+'pc_X_iter_'+str(ii)+'_hex_dim_' + str(i) + '_' + str(j) +'_basis_'+str(countBasis)+ '.png')
      #plt.show()
      plt.close()
    
  #for i in range(0,dimCoarse):
    #for j in range(i+1,dimCoarse):
      #fig = plt.figure(1)#, figsize=(1,1))
      #ax1 = fig.add_subplot(111)

      #smoothing = 200

      #heatmap_32, extent_32 = myplot(sPc[i,:],sPc[j,:], nb=smoothing)
      #ax1.imshow(heatmap_32, extent=extent_32, origin='lower', aspect='auto')
      ##ax1.set_title("Smoothing over 32 neighbors")
      ##plt.xlim([0,1])
      ##plt.ylim([0,1])
      #plt.title(r'$\mathbf{X} \sim p_c(\mathbf{X}|\theta),~\dim(\theta)='+str(countBasis)+'$')
      #plt.grid()
      ##plt.savefig('plots/pc_X_iter_'+str(ii)+'_samplehist_dim_' + str(i) + '_' + str(j) +'_basis_'+str(countBasis)+ fFormat)
      #plt.close()
      ##plt.show()

quit()
