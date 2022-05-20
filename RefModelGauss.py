# -*- coding: utf-8 -*
from __future__ import unicode_literals

import os
import scipy
import numpy as np
import torch
import math
import torch.nn.functional as F

from SchedulerBeta import SchedulerBeta
from SchedulerBeta import SchedulerBetaKL

import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
font = {'weight': 'normal',
        'size': 16}
# font = {'weight' : 'normal',
#        'size'   : 5}
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


class ReferenceModelGauss:
    def __init__(self):
        pass

    def doStep(self):
        pass

    def plotencodedgaussian(self, vaemodel, samples, p_val, z_rep, path, postfix='', name_colormap='viridis',
                            mcomponents=1):

        if vaemodel.x_dim == 2:

            f, ax = plt.subplots(1, 2)

            # set axis in right subplot to the right
            ax[1].yaxis.tick_right()
            ax[1].yaxis.set_label_position('right')

            pmin = p_val.min()
            pmax = p_val.max()
            pscaled = (p_val - pmin) / (pmax - pmin)

            nsamples = samples.shape[0]
            npercomonent = int(nsamples / mcomponents)
            componentarray = np.zeros(nsamples, dtype=int)

            if mcomponents > 1:
                cm = plt.cm.get_cmap('cubehelix', mcomponents + 1)

                for i in range(mcomponents):
                    componentarray[i * npercomonent:(i + 1) * npercomonent] = i
                cmap = cm(componentarray)

                # pscaledarray = pscaled.reshape(nsamples, 1)
                # colorstot = np.concatenate((cmap, pscaledarray), axis=1)
                cmap[:, 3] = pscaled

                ax[0].scatter(samples[:, 0], samples[:, 1], color=cmap, s=5)
                ax[0].set_xlim([-1.5, 1.5])
                ax[0].set_ylim([-1.5, 1.5])
            else:
                cm = getattr(matplotlib.cm, name_colormap)
                cmap = cm(pscaled)

                ax[0].scatter(samples[:, 0], samples[:, 1], c=cmap, s=5)
            ax[0].set_title(r'$x \sim p_{\text{target}}(\mathbf{x})$')

            if vaemodel.z_dim == 1:

                # This is just to separate the encoded samples by its component.
                if mcomponents == 1:
                    z_x = np.zeros(z_rep.shape) - 0.075
                elif mcomponents > 1:
                    z_x = (componentarray.astype('float') - 1.5) / 20.
                else:
                    raise ValueError('Amount of mixture components are not properly defined.')

                ax[1].scatter(z_rep, z_x, c=cmap, s=5)
                ax[1].hist(z_rep, alpha=0.5, density=True, label='Histogram of encoded x')
                leghist = mpatches.Patch(color='C0', alpha=0.5, label='Histogram of encoded x')
                ax[1].set_ylabel('Encoded samples from component number')
                # ax[1].set_ylim([-1, 10])
                ax[1].set_ylim([-0.1, 0.45])
                ax[1].yaxis.set_ticks([])
                ax[1].yaxis.set_ticks(np.array([-0.075, -0.025]))
                ax[1].set_yticklabels(['0', '1'])
                ax[1].set_xlim([-3, 3])

                xaxis = np.linspace(-3, 3, 301)

                # This would plot a second y-axis
                # ax1second = ax[1].twinx()
                # legprior, = ax1second.plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3', label=r'$p(z) = N(0,1)$')

                legprior, = ax[1].plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3',
                                       label=r'$q(\mathbf{z}) = \mathcal{N}(0,1)$')
                ax[1].axhline(y=0, xmin=xaxis[0], xmax=xaxis[-1], ls='--', c='k', alpha=0.5)

                ax[1].legend(handles=[leghist, legprior], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1)
                ax[1].set_title(r'Latent Representation $\dim(\mathbf{z})=1$')


            else:
                ax[1].scatter(z_rep[:, 0], z_rep[:, 1], c=cmap, s=5)
                ax[1].set_title('Latent Representation')

            f.savefig(path + '/latent_rep' + postfix + '.png', bbox_inches='tight')

            plt.close(f)

        elif (vaemodel.x_dim > 2) and (vaemodel.z_dim <= 2):

            f, ax = plt.subplots(1)

            pmin = p_val.min()
            pmax = p_val.max()
            pscaled = (p_val - pmin) / (pmax - pmin)

            nsamples = samples.shape[0]
            npercomonent = nsamples / mcomponents
            componentarray = np.zeros(nsamples, dtype=int)

            if mcomponents > 1:
                cm = plt.cm.get_cmap('cubehelix', mcomponents + 1)

                for i in range(mcomponents):
                    componentarray[i * npercomonent:(i + 1) * npercomonent] = i
                cmap = cm(componentarray)

                # pscaledarray = pscaled.reshape(nsamples, 1)
                # colorstot = np.concatenate((cmap, pscaledarray), axis=1)
                cmap[:, 3] = pscaled
            else:
                cm = getattr(matplotlib.cm, name_colormap)
                cmap = cm(pscaled)

            if vaemodel.z_dim == 1:

                if mcomponents == 1:
                    z_x = np.zeros(z_rep.shape)
                elif mcomponents > 1:
                    nsamples = samples.shape[0]
                    npercomonent = nsamples / mcomponents
                    z_x = (componentarray.astype('float') - 1.5) / 20.
                    # z_x = componentarray

                ax.scatter(z_rep, z_x, c=cmap, s=5)
                ax.hist(z_rep, alpha=0.5, density=True, label='Histogram of encoded x')
                leghist = mpatches.Patch(color='C0', alpha=0.5, label='Histogram of encoded x')
                ax.set_ylabel('Encoded samples from component #')
                # ax[1].set_ylim([-1, 10])
                ax.set_ylim([-0.1, 0.45])
                ax.yaxis.set_ticks([])
                ax.yaxis.set_ticks(np.array([-0.075, -0.025]))
                ax.set_yticklabels(['0', '1'])
                ax.set_xlim([-3, 3])

                xaxis = np.linspace(-3, 3, 301)

                # This would plot a second y-axis
                # ax1second = ax[1].twinx()
                # legprior, = ax1second.plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3', label=r'$p(z) = N(0,1)$')

                legprior, = ax.plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3', label=r'$p(z) = N(0,1)$')
                ax.axhline(y=0, xmin=xaxis[0], xmax=xaxis[-1], ls='--', c='k', alpha=0.5)

                ax.legend(handles=[leghist, legprior], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
                ax.set_title(r'Latent Representation $\dim(z)=1$')

            else:
                ax.scatter(z_rep[:, 0], z_rep[:, 1], c=cmap, s=5)
                ax.set_title('Latent Representation')

            f.savefig(path + '/latent_rep' + postfix + '.png', bbox_inches='tight')

            plt.close(f)


class ReferenceModelSingleGauss(ReferenceModelGauss):
    def __init__(self, mu, sigma, W, outputdir, bgpu=False, bstepsched=False):
        super(ReferenceModelSingleGauss, self).__init__()

        #self.mixtures = mu.shape[0]
        self.dim = mu.shape[0]

        self.npmu = mu.numpy()
        self.npsigma = sigma.numpy()
        self.npW = W.numpy()
        Wexp = np.expand_dims(self.npW, 0)
        self.np_cov = Wexp * Wexp.T + np.diag(np.power(self.npsigma, 2))

        self.rank = np.linalg.matrix_rank(self.np_cov)

        print(np.linalg.slogdet(self.np_cov))

        self.np_cov_inv = np.linalg.inv(self.np_cov)

        np.savetxt(outputdir + '/mu_ref.txt', self.npmu)
        np.savetxt(outputdir + '/mu_sigs.txt', self.np_cov.flatten())

        # define variables
        if bgpu:
            with torch.no_grad():
                self.vmu = mu.cuda()
                temp1 = torch.from_numpy(self.np_cov)
                self.vcov = temp1.cuda()
                temp2 = torch.from_numpy(self.np_cov_inv)
                self.vcovinv = temp2.cuda()
                self.vsig = sigma.cuda()
        else:
            with torch.no_grad():
                self.vmu = mu
                self.vcov = torch.from_numpy(self.np_cov)
                self.vcovinv = torch.from_numpy(self.np_cov_inv)
                self.vsig = sigma

        self.mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.vmu, covariance_matrix=self.vcov)

        self.refsamples = self.sample_np(10000).T

    def logpx(self, x, bgpu=False, bfreememory=False):
        bcov = True

        muref = self.getmu()
        covinvref = self.getInvCov()
        covref = self.getCov()
        x_dim_mod = x.shape[1]
        N = x.shape[0]

        if bgpu:
            murefexpanded = muref.expand(x.shape[0], x.shape[1]).cuda()
        else:
            murefexpanded = muref.expand(x.shape[0], x.shape[1])

        if bcov:
            xmmu = x - murefexpanded
            if bfreememory:
                del murefexpanded

            xmmucovinv = torch.mm(xmmu, covinvref)
            xcovinvx = torch.mul(xmmucovinv, xmmu)
            logpx = - 0.5 * xcovinvx.sum()
            logvarpx = covref.det().log()
        else:
            pointwiseLogpx = -0.5 * F.mse_loss(x, murefexpanded, size_average=False, reduce=False)
            if bfreememory:
                del murefexpanded
            pxSigma = self.getSigma()
            sgima = pxSigma
            if bgpu:
                sgima = sgima.cuda()
            sigsq = torch.mul(sgima, sgima)

            if bgpu:
                weight = sigsq.reciprocal().cuda()  # 1./sigsq
            else:
                weight = sigsq.reciprocal()  # 1./sigsq
            weightexpanded = weight.expand(x.shape[0], x.shape[1])

            # logvarobjective = 0.5 * recon_logvar.sum()

            pointwiseWeightedMSEloss = pointwiseLogpx.mul(weightexpanded)
            if bfreememory:
                del pointwiseWeightedMSEloss, weightexpanded, weight

            logpx = pointwiseWeightedMSEloss.sum()  # xcovinvx.sum()
            logvarpx = sigsq.log().sum()

        logpx -= 0.5 * logvarpx * N
        if bgpu:
            logpx -= x_dim_mod * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N
        else:
            logpx -= x_dim_mod * 0.5 * torch.tensor(2 * math.pi).log() * N

        return logpx

    def getNPmu(self):
        return self.npmu

    def getNPsigs(self):
        return self.np_cov.flatten()

    # return log probability
    def logprob(self, x):
        return self.logprob(x)

    # return mean of mvn
    def getmu(self):
        return self.vmu

    def getIsMixture(self):
        return False

    def getModelType(self):
        return 'Gaussian'

    def getMixtures(self):
        return 1

    # return Sigma of mvn
    def getSigma(self):
        return self.vsig

    def getCov(self):
        return self.vcov

    def getInvCov(self):
        return self.vcovinv

    def getCovNP(self):
        return self.np_cov

    def getInvCovNP(self):
        return self.np_cov_inv

    def sample_np(self, nsamples):
        return np.random.multivariate_normal(mean=self.npmu, cov=self.np_cov, size=nsamples)

    def sample_torch(self, nsamples):
        return torch.from_numpy(np.random.multivariate_normal(mean=self.npmu, cov=self.np_cov, size=nsamples).astype(dtype='float32'))

    def getpdf(self, samples):
        return scipy.stats.multivariate_normal.pdf(samples, mean=self.npmu, cov=self.np_cov)

    def getlogpdfTorch(self, samplesTorch):
        return self.mvn.log_prob(samplesTorch)

    def getRefSamples(self):
        return self.refsamples

    def doStep(self):
        pass

    # plot the distribution if 2D
    def plot(self, path=''):
        a = 1
        if self.dim == 2:
            from utils import plotmvn
            mu = self.vmu.cpu().numpy()
            Sigma = self.vcov.cpu().numpy()
            plotmvn(fname=os.path.join(path, 'plot_mvntest.pdf'), mu=mu, Sigma=Sigma)


class ReferenceModelMultiModal(ReferenceModelGauss):
    def __init__(self, mu, sigma, W, outputdir, bgpu=False, stepschedopt=None):
        super(ReferenceModelMultiModal, self).__init__()

        if not (stepschedopt or stepschedopt['usestepsched']):
            pass
        else:
            offset = 20
            self.schedBeta = SchedulerBeta(a_init=1.e-7, a_end=1., checknlast=4, avginterval=10,
                                           expoffset=offset, outputpath=outputdir,
                                           angular=False, bLin=True, maxsteps=stepschedopt['imaxstepssched'],
                                           bresetopt=stepschedopt['stepschedresetopt'])

        self.mixtures = mu.shape[0]
        self.dim = mu.shape[1]

        self.npmu = mu.numpy()
        self.npsigma = sigma.numpy()
        self.npW = W.numpy()

        self.np_cov_list = []
        self.np_cov_inv_list = []
        for i in range(0, self.mixtures):
            #self.np_cov_list.append(np.outer(self.npW[i, :], self.npW[i, :].T) + np.diag(np.power(self.npsigma[i, :], 2)))
            Wexp = np.expand_dims(self.npW[i, :], 0)
            self.np_cov_list.append(Wexp * Wexp.T + np.diag(np.power(self.npsigma[i, :], 2)))
            self.np_cov_inv_list.append(np.linalg.inv(self.np_cov_list[i]))

            self.rank = np.linalg.matrix_rank(self.np_cov_list[i])
            print(np.linalg.slogdet(self.np_cov_list[i]))
            np.savetxt(outputdir + '/mu_sigs_'+str(i)+'.txt', self.np_cov_list[i].flatten())

        np.savetxt(outputdir + '/mu_ref.txt', self.npmu)

        # define variables
        if bgpu:
            with torch.no_grad():
                self.vmu = mu.cuda()
                self.vcov = torch.zeros([self.mixtures, self.dim, self.dim]).cuda()
                self.vcovinv = torch.zeros([self.mixtures, self.dim, self.dim]).cuda()
                self.vsig = sigma.cuda()
                for i in range(0, self.mixtures):
                    temp1 = torch.from_numpy(self.np_cov_list[i])
                    self.vcov[i, :, :] = temp1.cuda()
                    temp2 = torch.from_numpy(self.np_cov_inv_list[i])
                    self.vcovinv[i, :, :] = temp2.cuda()
        else:
            with torch.no_grad():
                self.vmu = mu
                self.vcov = torch.zeros([self.mixtures, self.dim, self.dim])
                self.vcovinv = torch.zeros([self.mixtures, self.dim, self.dim])
                self.vsig = sigma
                for i in range(0, self.mixtures):
                    self.vcov[i, :, :] = torch.from_numpy(self.np_cov_list[i])
                    self.vcovinv[i, :, :] = torch.from_numpy(self.np_cov_inv_list[i])

        # TOFIX !!!!
        self.refsamples = self.sample_np(10000).T

        self.mvn_components_list = []
        for i in range(0, self.mixtures):
            self.mvn_components_list.append(torch.distributions.multivariate_normal.MultivariateNormal(loc=self.vmu[i, :], precision_matrix=self.vcovinv[i, :, :]))
        #self.mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.vmu, covariance_matrix=self.vcov)

    def logpx(self, x, bgpu=False, bfreememory=False):

        mixtureweight = 0.5
        nmixture = self.getMixtures()
        muref = self.getmu()
        x_dim_mod = x.shape[1]

        if bgpu:
            x_i = torch.zeros(x.shape[0], nmixture).cuda()
        else:
            x_i = torch.zeros(x.shape[0], nmixture)

        for i in range(0, nmixture):

            covinvref = self.getInvCov(i)
            covref = self.getCov(i)

            if bgpu:
                murefexpanded = muref[i, :].expand(x.shape[0], x.shape[1]).cuda()
            else:
                murefexpanded = muref[i, :].expand(x.shape[0], x.shape[1])

            xmmu = x - murefexpanded
            if bfreememory:
                del murefexpanded

            xmmucovinv = torch.mm(xmmu, covinvref)
            xcovinvx = torch.sum(torch.mul(xmmucovinv, xmmu), dim=1)

            log2pi = np.log(2 * math.pi)

            x_i[:, i] = xcovinvx.mul(-0.5).add(-0.5 * x_dim_mod * log2pi + np.log(mixtureweight))

            # TODO Check if there is covref.logdet() available. This could cause numerical instabilities.
            x_i[:, i] -= 0.5 * covref.det().log()

            if bfreememory:
                del xcovinvx

            # if bgpu:
            #    x_i[i] -= self.x_dim_mod * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
            # else:
            #    x_i[i] -= self.x_dim_mod * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx

            # x_i[i] -= 0.5 * covref.det().log()

        m, m_pos = x_i.max(dim=1, keepdim=True)
        xma = x_i - m
        expxma = torch.exp(xma)
        sumexpxma = expxma.sum(dim=1, keepdim=True)
        logsumtemp = torch.log(sumexpxma)
        logsumtemppm = m + logsumtemp
        logpx = logsumtemppm.sum()

        if hasattr(self, 'schedBeta'):
            beta = self.schedBeta.getLearningPrefactor()
            logpx *= beta

        return logpx

    def doStep(self):
        if hasattr(self, 'schedBeta'):
            self.schedBeta.doStep()

    def getNPmu(self):
        return self.npmu

    def getNPsigs(self):
        return self.np_cov.flatten()

    # return log probability
    def logprob(self, x):
        return self.logprob(x)

    # return mean of mvn
    def getmu(self):
        return self.vmu

    def getIsMixture(self):
        return True

    def getModelType(self):
        return 'GaussianMixture'

    def getMixtures(self):
        return self.mixtures

    # return Sigma of mvn
    def getSigma(self):
        return self.vsig

    def getCov(self, component):
        return self.vcov[component, :, :]

    def getInvCov(self, component):
        return self.vcovinv[component, :, :]

    def getCovNP(self, component):
        return self.np_cov_list[component]

    def getInvCovNP(self, component):
        return self.np_cov_inv_list[component]

    def sample_np(self, nsamples):
        nsamplespermixture = int(nsamples/self.mixtures)
        samples = np.zeros([nsamples, self.dim])
        for i in range(0, self.mixtures):
            samples[(i * nsamplespermixture):((i + 1) * nsamplespermixture), :] = np.random.multivariate_normal(mean=self.npmu[i,:], cov=self.np_cov_list[i], size=nsamplespermixture)
        return samples

    def sample_torch(self, nsamples):
        nsamplespermixture = int(nsamples/self.mixtures)
        samples = np.zeros([nsamples, self.dim])
        for i in range(0, self.mixtures):
            samples[i * nsamplespermixture:(i + 1) * nsamplespermixture, :] = np.random.multivariate_normal(mean=self.npmu[i, :], cov=self.np_cov_list[i], size=nsamplespermixture)
        return torch.from_numpy(samples.astype(dtype='float32'))

    def getpdf(self, samples):
        #print 'Implement getpdf in reference model.'
        fraction = 1./self.mixtures
        pdf = np.zeros(samples.shape[0])
        for i in range(0, self.mixtures):
            pdf += fraction * scipy.stats.multivariate_normal.pdf(samples, mean=self.npmu[i, :], cov=self.np_cov_list[i])
        return pdf

    def getlogpdfTorch(self, samplesTorch):
        #print 'Implement getpdf in reference model.'
        fraction = 1./self.mixtures
        logpdf = torch.zeros(samplesTorch.shape[0], self.mixtures)

        for i in range(0, self.mixtures):
            # from pdf
            logpdf[:, i] = self.mvn_components_list[i].log_prob(samplesTorch)
            # from fraction
            logpdf[:, i] = logpdf[:, i].add(np.log(fraction))

        m, m_pos = logpdf.max(dim=1, keepdim=True)
        xma = logpdf - m
        expxma = torch.exp(xma)
        sumexpxma = expxma.sum(dim=1, keepdim=True)
        logsumtemp = torch.log(sumexpxma)
        logsumtemppm = m + logsumtemp

        return logsumtemppm

    def getRefSamples(self):
        return self.refsamples

    # plot the distribution if 2D
    def plot(self, path):
        refsamples = self.refsamples
        if refsamples is not None:
            f, ax = plt.subplots(1)  # , tight_layout=True)
            # #f.suptitle(r'Prediction: $\mu_1$ = %1.3f  $\mu_2$ = %1.3f $\sigma_1^2$ = %1.3 $\sigma_2^2$ = %1.3' %(mean[0], mean[1], std[0], std[1]) )
            counts, xedges, yedges, im = ax.hist2d(refsamples[0, :], refsamples[1, :], bins=[80, 80], norm=colors.LogNorm(vmin=1, vmax=200))
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1.5, 1.5])
            f.colorbar(im)

            f.savefig(path + '/reference.png', bbox_inches='tight')#, pad_inches=0)
            plt.close(f)

        #a = 1
        #if self.dim == 2:
        #from utils import plotmvn
        #mu = self.vmu.numpy()
        #Sigma = self.vsig.numpy()
        #plotmvn(fname='plot_mvntest.pdf', mu=mu, Sigma=Sigma)


class EnergyFunctionalNoe:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.sine_add = True
        e_max = 25.  # E.max()
        self.sine_mag = e_max / 5.
        self.f = (1. / (2 * np.pi)) * 120

    def evaluate(self, x):

        # unsqueeze if a single poit is evaluated
        if x.dim() == 1:
            xu = x.unsqueeze(0)
        else:
            xu = x

        E = 0.25 * self.a * torch.pow(xu[:, 0], 4.) - 0.5 * self.b * torch.pow(xu[:, 0], 2.)\
            + self.c * xu[:, 0] + 0.5 * self.d * torch.pow(xu[:, 1], 2.)

        #if self.sine_add:
        #    E += self.sine_mag * torch.sin(self.f * xu[:, 0])

        return E


class ReferenceModelEnergyFunctional:
    def __init__(self, outputdir, bgpu=False, stepschedopt=None, ref_sampling='imp'):#bstepsched=False, maxschedsteps=800):
        if not (stepschedopt is None or stepschedopt['usestepsched']):
            raise Warning('No scheduler used.')
        else:
            if stepschedopt['stepschedtype'] in 'lin':
                offset = 20
                self.schedBeta = SchedulerBeta(a_init=1.e-3, a_end=1., checknlast=4, avginterval=10, expoffset=offset,
                                               outputpath=outputdir, angular=False, bLin=True,
                                               maxsteps=stepschedopt['imaxstepssched'],
                                               bresetopt=stepschedopt['stepschedresetopt'], intwidth=stepschedopt['stepschedconvcrit'])
            elif stepschedopt['stepschedtype'] in 'kl':
                self.schedBeta = SchedulerBetaKL(a_init=1.e-4,
                #self.schedBeta = SchedulerBetaKL(a_init=1.e-7,
                                                 #a_init=1.,
                                                 a_end=1.,
                                                 outputpath=outputdir,
                                                 bresetopt=stepschedopt['stepschedresetopt'], max_beta_increase=1.e-3)
                if stepschedopt is not None:
                    self.schedBeta.setSched(stepschedopt['usestepsched'])
            else:
                raise NotImplementedError('Scheduler option not implemented.')

        a, b, c, d = 1., 6., 1., 1.
        #a, b, c, d = 1., 6., 0., 1.

        self.E = EnergyFunctionalNoe(a, b, c, d)
        self.x_dim = 2
        self.benforcebeta = True
        self.bgpu = bgpu
        if bgpu:
            self.device = torch.device('cuda')
            nref = 100000
        else:
            self.device = torch.device('cpu')
            nref = 1000

        if ref_sampling is 'imp':
            from MCsampler import ImportanceSampling
            self.importsamp = ImportanceSampling(p_distribution=self, dim=self.x_dim, device=self.device)
            #self.refsamples, self.logw = self.importsamp.sample(nsamples=100000, breweight=False)
            self.refsamples = self.importsamp.sample(nsamples=nref)

            self.plot(path=outputdir, file_name='reference_imp.png', samples_given=self.refsamples)
            self.plot_reference(samples=self.refsamples, outputpath=outputdir)

            #mean = self.importsamp.mean()
            mean = self.refsamples.mean(dim=0)
            std = self.refsamples.mean(dim=0)
            # store reference statistics
            np.savetxt(os.path.join(outputdir, 'mean_ref.txt'), mean.data.cpu().numpy())
            np.savetxt(os.path.join(outputdir, 'std_ref.txt'), std.data.cpu().numpy())

        else:
            from MCsampler import MetropolisHastings
            # create a MCMC object
            self.mcmc = MetropolisHastings(distribution=self, dim=self.x_dim, sigmaSq=0.01, bgpu=self.bgpu)
            self.refsamples = self.mcmc.sample(nsamples=nref, beta=1.)
            mean = self.refsamples_mcmc.mean(dim=0)
            self.plot_x_axis_frequency(self.refsamples, outputdir)
            self.plot(path=outputdir, file_name='reference_mcm.png', samples_given=self.refsamples_mcmc)


    def plot_x_axis_frequency(self, samples, path):

        f, ax = plt.subplots()

        ax.plot(samples[:, 0].data.cpu().numpy())
        ax.set_xlabel('Steps')
        ax.set_ylabel(r'$x_1$')

        f.savefig(os.path.join(path, 'ref_x.png'), bbox_inches='tight')
        plt.close(f)

    def setenforcebeta(self, bvalue):
        raise DeprecationWarning('This should be not used anymore since we enforce beta now differently.')
        self.benforcebeta = bvalue

    def logpx(self, x, beta=None, bsum=True, bgpu=False, bfreememory=False):

        E = self.E.evaluate(x)

        if bsum:
            U = E.sum()
        else:
            U = E
        logpx = -U

        #if hasattr(self, 'schedBeta') and not self.benforcebeta:
        #    betasched = self.schedBeta.getLearningPrefactor()
        #    logpx *= betasched
        #else:
        #    logpx *= beta

        if hasattr(self, 'schedBeta') and beta is None:
            beta = self.schedBeta.getLearningPrefactor()
        else:
            beta = 1. if beta is None else beta
        logpx *= beta

        return logpx

    def doStep(self):
        if hasattr(self, 'schedBeta'):
            self.schedBeta.doStep()

    # return log probability
    def logprob(self, x, beta=1.):
        raise ValueError('can I remove logprob function of referencemodel?')
        quit()

    def getIsMixture(self):
        return False

    def getModelType(self):
        return 'Quad'

    def sample_np(self, nsamples, beta=1.):
        samples_torch = self.sample_torch(nsamples=nsamples, beta=beta)
        if self.bgpu:
            samples_np = samples_torch.data.cpu().numpy()
        else:
            samples_np = samples_torch.data.numpy()
        return samples_np

    def sample_torch(self, nsamples, beta=1.):
        with torch.no_grad():
            samples = self.mcmc.sample(nsamples=nsamples, beta=beta)
        return samples

    def getpdf(self, samples, beta=1.):
        E = self.E.evaluate(samples)
        logpx = -E
        logpx *= beta
        return logpx.exp()

    def getlogpdfTorch(self, samplesTorch, beta=1.):
        logpdf = self.logpx(x=samplesTorch, beta=beta, bsum=False)
        #raise ValueError('getlogpdfTorch should be replaced by logpx with reduce=\'none\'.')
        return logpdf

    def getRefSamples(self, nsubsamples=None):
        if nsubsamples is None:
            return self.refsamples
        else:
            return self.refsamples[0:-1:10, :]

    def getRefWeights(self, nsubsamples=None):
        if nsubsamples is None:
            if hasattr(self, 'logw'):
                return self.logw.exp()
            else:
                return None
        else:
            if hasattr(self, 'logw'):
                return self.logw[0:-1:10, :].exp()
            else:
                return None

    # plot the distribution if 2D
    def plot(self, path, file_name='reference.png', samples_given=None):
        if samples_given is None:
            if self.bgpu:
                refsamples = self.refsamples.cpu().data.numpy()
            else:
                refsamples = self.refsamples.data.numpy()
        else:
            if self.bgpu:
                refsamples = samples_given.data.cpu().numpy()
            else:
                refsamples = samples_given.data.numpy()

        f, ax = plt.subplots(1)  # , tight_layout=True)
        # #f.suptitle(r'Prediction: $\mu_1$ = %1.3f  $\mu_2$ = %1.3f $\sigma_1^2$ = %1.3 $\sigma_2^2$ = %1.3' %(mean[0], mean[1], std[0], std[1]) )
        #counts, xedges, yedges, im = ax.hist2d(refsamples[:, 0], refsamples[:, 1], bins=[80, 80], norm=colors.LogNorm(vmin=1, vmax=200))
        ax.grid(ls='dashed')
        ax.scatter(refsamples[:, 0], refsamples[:, 1])
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_axisbelow(True)
        #f.colorbar(im)
        print(path)
        f.savefig(os.path.join(path, file_name), bbox_inches='tight')#, pad_inches=0)
        plt.close(f)

    def plotpotentials(self, vaemodel, path, postfix='', beta=1.0, plot_betaU=True):

        # set model to evaluation mode
        vaemodel.eval()

        n = 300
        x = torch.zeros(n, self.x_dim, device=self.device)
        # set x1 the cut at x2 = 0
        x[:, 0] = torch.linspace(-4., 4., n, device=self.device)

        Upred = - vaemodel.getlogpdf(samples=x, bgpu=self.bgpu, nz=5000)
        Upred = Upred - Upred.min()

        Utarget = -self.getlogpdfTorch(samplesTorch=x)
        Utarget = Utarget - Utarget.min()

        xnp = x.cpu().data.numpy()
        Uprednp = Upred.cpu().data.numpy()
        Utargetnp = Utarget.cpu().data.numpy()

        f, ax = plt.subplots(1)
        ax.grid(ls='dashed')
        ax.plot(xnp[:, 0], Uprednp, label='Prediction', lw=2)
        ax.plot(xnp[:, 0], Utargetnp, label='Target', lw=2, ls='dashed')
        if plot_betaU:#beta != 1.0:
            betaUtarget = beta * Utarget
            betaUtargetnp = betaUtarget.cpu().data.numpy()
            ax.plot(xnp[:, 0], betaUtargetnp, label=r'Target $\beta = ${:.3}'.format(beta), lw=2, ls='dashdot')
        ax.set_ylim(-1.5, 35)
        ax.set_ylabel(r'$\beta U(x_1; x_2=0)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')
        f.savefig(path + '/pot' + postfix + '.png', bbox_inches='tight')
        plt.close(f)

    def vis_latentpredictions(self, vaemodel, path, postfix='', do_latent_movement=False):

        vaemodel.eval()
        vaemodel.bgetlogvar = True

        vis_mean = False

        #from SetPlotFont import loadfontsetting as setfonts
        #setfonts()

        if vis_mean:
            z = torch.linspace(-2, 2, 200, device=self.device)
            x, logvar = vaemodel.decode(z.unsqueeze(1))

            z_np = z.data.cpu().numpy()
            z_np_min = z_np.min()
            z_np_max = z_np.max()
            z_np_norm = (z_np - z_np_min)/(z_np_max - z_np_min)
            x_np = x.data.cpu().numpy()

            cm = plt.cm.get_cmap('viridis')
            cmap = cm(z_np_norm)

            f, ax = plt.subplots(1)
            ax.scatter(x_np[:, 0], x_np[:, 1], c=cmap)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-5, 5)
            f.savefig(path + '/latent_rep_given_z' + postfix + '.png', bbox_inches='tight')
            plt.close(f)

        z = torch.randn(2000, 1, device=self.device)
        xmu, logvar = vaemodel.decode(z)
        x_sample = xmu
        sigma = torch.sqrt(logvar.exp())
        for i in range(5):
            x_sample = torch.cat((x_sample, torch.randn_like(xmu) * sigma + xmu), 0)
        zmu_encoded, zlogvar_encoded = vaemodel.encode(x_sample)
        z_np = zmu_encoded.data.cpu().numpy()
        z_np = z_np.squeeze()
        z_np_min = z_np.min()
        z_np_max = z_np.max()
        #z_np_norm = (z_np - z_np_min)/(z_np_max - z_np_min)
        z_np_min = -2.5
        z_np_max = 2.5
        z_np_norm = (z_np - z_np_min)/(z_np_max - z_np_min)
        x_np = x_sample.data.cpu().numpy()

        cm = plt.cm.get_cmap('viridis')
        cmap = cm(z_np_norm)

        f, ax = plt.subplots(1)
        ax.scatter(x_np[:, 0], x_np[:, 1], c=cmap)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-5, 5)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.grid(ls='dashed')
        ax.set_axisbelow(True)

        if False:
            f.subplots_adjust(right=0.8)
            cb_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])#([0.83, 0.1, 0.02, 0.8])
            norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
            cbar = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cm, norm=norm, orientation='vertical')
        else:
            divider = make_axes_locatable(ax)
            cb_ax = divider.new_vertical(size="7%", pad=0.6, pack_start=True)
            f.add_axes(cb_ax)
            norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
            cbar = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cm, norm=norm, orientation='horizontal')

        #cbar.ax.plot(0.5, mean, 'w.')  # my data is between 0 and 1
        # Marker on colorbar
        #cbar.ax.plot([0, 1], [0.5] * 2, 'w')  # my data is between 0 and 1

        #cbar = f.colorbar(z_np_norm, ax=ax, pad=-0.44, shrink=0.8, boundaries=[-z_np_min, z_np_max], format='%r°')
        # cbar = f.colorbar(cs1temp, ax=ax, pad=-0.44, shrink=0.8, ticks=[-180, -135, -90, -45, 0, 45, 90, 135, 180], boundaries=[-180, 180], format=r'$%r^\circ$')
        # cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi /2$', r'$\pi$'])
        # cbar.ax.set_title(r'$%s [^\circ]$'% bar_name)
        #cbar.ax.set_title(r'$z$')
        cbar.ax.set_ylabel(r'$z$', rotation='horizontal', labelpad=10)
        #clab = cbar.ax.get_label()
        #clab.set_verticalalignment('center')
        #clab.set_horizontalalignment('left')

        f.savefig(path + '/latent_encoded_x' + postfix + '.png', bbox_inches='tight')
        plt.close(f)

        if do_latent_movement:
            max_z_val = np.linspace(z_np_min, z_np_max, 30)
            outputpath = os.path.join(path, 'latent' + postfix)
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
            for idx, zmax in enumerate(max_z_val):
                z_np_max_norm = (zmax - z_np_min) / (z_np_max - z_np_min)
                z_cond_idx = np.where(z_np_norm < z_np_max_norm)[0]
                if True: #z_cond_idx.shape[0] > 0:
                    x_cond = x_np[z_cond_idx, :]
                    col_cond = cmap[z_cond_idx, :]

                    f, ax = plt.subplots(1)
                    ax.scatter(x_cond[:, 0], x_cond[:, 1], c=col_cond)
                    ax.set_xlabel(r'$x_1$')
                    ax.set_ylabel(r'$x_2$')
                    ax.set_ylim(-5, 5)
                    ax.set_xlim(-4, 4)
                    ax.grid(ls='dashed')
                    ax.set_axisbelow(True)

                    if False:
                        f.subplots_adjust(right=0.8)
                        cb_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])  # ([0.83, 0.1, 0.02, 0.8])
                        norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
                        cbar = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cm, norm=norm, orientation='vertical')
                        # cbar.ax.plot(0.5, mean, 'w.')  # my data is between 0 and 1
                        # Marker on colorbar
                        xlim_cbar = cbar.ax.get_xlim()
                        cbar.ax.plot([xlim_cbar[0], xlim_cbar[1]], [zmax] * 2, 'w', lw=5)  # my data is between 0 and 1
                    else:
                        divider = make_axes_locatable(ax)
                        cb_ax = divider.new_vertical(size="7%", pad=0.6, pack_start=True)
                        f.add_axes(cb_ax)
                        norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
                        cbar = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cm, norm=norm, orientation='horizontal')
                        ylim_cbar = cbar.ax.get_ylim()
                        cbar.ax.plot([zmax] * 2, [ylim_cbar[0], ylim_cbar[1]], 'w', lw=4)


                    #cbar.ax.set_title(r'$z$')
                    cbar.ax.set_ylabel(r'$z$', rotation='horizontal', labelpad=10)

                    f.savefig(outputpath + '/latent_encoded_x' + postfix + '_' + str(idx) + '.png', bbox_inches='tight')
                    plt.close(f)

    def plot_reference(self, samples, outputpath):
        refsamples = samples.data.cpu().numpy()

        settingenergyfunctional = {'setlim': True, 'nbins': 40, 'vmin': 0.0001, 'vmax': 1.}
        setting = settingenergyfunctional
        nbins = setting['nbins']
        vmin = setting['vmin']
        vmax = setting['vmax']

        f, ax = plt.subplots()  # , tight_layout=True)
        # #f.suptitle(r'Prediction: $\mu_1$ = %1.3f  $\mu_2$ = %1.3f $\sigma_1^2$ = %1.3 $\sigma_2^2$ = %1.3' %(mean[0], mean[1], std[0], std[1]) )
        countsref, xedgesref, yedgesref, imref = ax.hist2d(refsamples[:, 0], refsamples[:, 1], bins=[nbins, nbins], normed=True,
                  norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.set_ylim(-5., 5.)
        ax.set_xlim(-4., 4.)
        ax.set_ylabel(r'$x_2$')
        ax.set_xlabel(r'$x_1$')
        ax.grid(ls='dashed')
        ax.set_axisbelow(True)
        f.colorbar(imref)
        f.savefig(outputpath + '/reference_eng.png', bbox_inches='tight')  # , pad_inches=0)
        plt.close(f)

    def plotencodedrepresentaion(self, samples, p_val, z_rep, path, postfix='', name_colormap='viridis',
                            mcomponents=2):
        z_dim = z_rep.shape[1]

        from SetPlotFont import loadfontsetting as setfonts
        setfonts()

        if self.x_dim == 2:

            f, ax = plt.subplots(1, 2)

            # Set axis in right subplot to the right
            ax[1].yaxis.tick_right()
            ax[1].yaxis.set_label_position('right')

            # For obtaining the 0-1 representation and assigning colors.
            pmin = p_val.min()
            pmax = p_val.max()
            pscaled = (p_val - pmin) / (pmax - pmin)

            nsamples = samples.shape[0]
            # Assign component according x-value
            #### IF SAMPLES IS TORCH TENSOR
            # componentarraytorch = samples[:, 0] > 0.
            # componentarray = componentarraytorch.data.numpy()

            #### IF SAMPLES IS NUMPY ARRAY
            componentarray = (samples[:, 0] > 0).astype(int)

            #componentarray = componentarraytorch.data.numpy()
            cm = plt.cm.get_cmap('cubehelix', 3)
            cmap = cm(componentarray)

            #pscaledarray = pscaled.reshape(nsamples, 1)
            #colorstot = np.concatenate((cmap, pscaledarray), axis=1)
            cmap[:, 3] = pscaled

            ax[0].scatter(samples[:, 0], samples[:, 1], color=cmap, s=5)
            ax[0].set_xlim([-5, 5])
            ax[0].set_ylim([-4, 4])
            ax[0].set_xlabel(r'$x_1$')
            ax[0].set_ylabel(r'$x_2$')
            ax[0].set_title(r'$x \sim p_{\text{target}}(\mathbf{x})$')

            if z_dim == 1:

                # This is just to separate the encoded samples by its component.
                if mcomponents == 1:
                    z_x = np.zeros(z_rep.shape) - 0.075
                elif mcomponents > 1:
                    z_x = (componentarray.astype('float') - 1.5)/20.

                ax[1].scatter(z_rep, z_x, c=cmap, s=5)
                ax[1].hist(z_rep, alpha=0.5, density=True, label='Histogram of encoded x')
                leghist = mpatches.Patch(color='C0', alpha=0.5, label='Histogram of encoded x')
                ax[1].set_ylabel('Encoded samples from mode')
                #ax[1].set_ylim([-1, 10])
                ax[1].set_ylim([-0.1, 0.45])
                ax[1].yaxis.set_ticks([])
                ax[1].yaxis.set_ticks(np.array([-0.075, -0.025]))
                ax[1].set_yticklabels(['0', '1'])
                ax[1].set_xlim([-3, 3])
                ax[1].set_xlabel(r'z')

                xaxis = np.linspace(-3, 3, 301)

                # This would plot a second y-axis
                #ax1second = ax[1].twinx()
                #legprior, = ax1second.plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3', label=r'$p(z) = N(0,1)$')

                legprior, = ax[1].plot(xaxis, scipy.stats.norm.pdf(xaxis), c='C3', label=r'$q(\mathbf{z}) = \mathcal{N}(0,1)$')
                ax[1].axhline(y=0, xmin=xaxis[0], xmax=xaxis[-1], ls='--', c='k', alpha=0.5)

                ax[1].legend(handles=[leghist, legprior], loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1)
                ax[1].set_title(r'Latent Representation $\dim(\mathbf{z})=1$')
            else:
                ax[1].scatter(z_rep[:, 0], z_rep[:, 1], c=cmap, s=5)
                ax[1].set_title('Latent Representation')

            f.savefig(path+'/latent_rep'+postfix + '.png', bbox_inches='tight')

            plt.close(f)

        else:
            raise ValueError('Plotting encoded versions of dim(x) > is not implemented yet.')


import unittest

class TestRefModelEnergy(unittest.TestCase):

    def setUp(self):
        self.wd = os.getcwd()
        self.refeng = ReferenceModelEnergyFunctional(outputdir=self.wd)

    def test_sample(self):
        samples = None
        try:
            samples = self.refeng.sample_torch(10)
        except:
            samples = None
        self.assertFalse(samples is None)

    def test_plot(self):
        bpassed = True
        try:
            beta = 1.
            samples = self.refeng.sample_torch(nsamples=10000, beta=beta)
            self.refeng.plot(path=self.wd, file_name='ref_beta_' + str(beta).replace('.', '') + '.png',
                             samples_given=samples)
            #for beta in np.linspace(0, 1., 5):
            #    samples = self.refeng.sample_torch(nsamples=1000, beta=beta)
            #    self.refeng.getlogpdfTorch(samples)
            #    self.refeng.plot(path=self.wd, file_name='ref_beta_' + str(beta).replace('.', '') + '.png', samples_given=samples)
        except:
            bpassed = False

        self.assertTrue(bpassed)