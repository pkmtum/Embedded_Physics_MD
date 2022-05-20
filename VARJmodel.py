from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import math

# due to plotting purposes
import numpy as np
import scipy
import scipy.stats

import matplotlib

matplotlib.use('Agg')

import matplotlib.patches as mpatches

import matplotlib


matplotlib.use('Agg')

font = {'weight' : 'normal',
        'size'   : 16}

#font = {'weight' : 'normal',
#        'size'   : 5}

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
from matplotlib import colors
import os


def colorpointsgaussian(x, nsamples, name_colmap=''):

    from scipy.stats import multivariate_normal
    x_dim = x.shape[1]
    var = multivariate_normal(mean=np.zeros(x_dim), cov=np.eye(x_dim))
    p = var.pdf(x)

    pmin = p.min()
    pmax = p.max()

    pscaled = (p - pmin) / (pmax - pmin)

    cm = getattr(matplotlib.cm, name_colmap)
    cmap = cm(pscaled)

    return cmap


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        #print(module)
        if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print(module)
            print('NaN gradient in ' + type(module).__name__)
            # set gradient to zero and continue with other iteration
            for idx, gi in enumerate(grad_input):
                if gi is not None:
                    if np.any(np.isnan(gi.data.cpu().numpy())):
                        print(idx)
                        grad_input[idx].mul_(0.)
                        print(grad_input[idx])

    model.apply(lambda module: module.register_backward_hook(check_grad))

class VAEparent(nn.Module):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VAEparent, self).__init__()

        if dropout_p <= 0:
            self.dropout_active = False
            self.dropout_p = None
            self.dropout_enc_dec = None
        else:
            self.dropout_active = True
            self.dropout_p = dropout_p
            self.dropout_enc_dec = dropout_enc_dec

        self.device = device
        self.log2pi = torch.tensor(2 * math.pi, device=self.device).log()

        self.bdebug = False
        self.bplotdecoder = False
        self.bplotencoder = False
        self.bgetlogvar = True

        self.bfixlogvar = bfixlogvar
        self.bfixenclogvar = bfixenclogvar

        self.x_dim = int(x_dim)
        self.z_dim = int(args.z_dim)

        mean = torch.zeros(self.z_dim, device=self.device)
        std = torch.ones(self.z_dim, device=self.device)
        self.qz = torch.distributions.normal.Normal(loc=mean, scale=std)

        self.listenc = []
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        #self.prelu = nn.PReLU()
        self.celu = nn.CELU()

        self.grad_norm_storage_list_names_stored = False
        self.grad_norm_storage_list = list()
        #self.grad_norm_tot = list()

        self.np_array_tot_grad = None

        if self.bdebug:
            register_nan_checks(self)

    def getExclParamList(self, listExclNames, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if name not in listExclNames:
                yield param

    def getNamedParamList(self, listInclNames, recurse=True):
        for parname, param in self.named_parameters(recurse=recurse):
            for nameitemtocheck in listInclNames:
                if nameitemtocheck in parname:
                    yield param

    def setRequiresGrad(self, network_prefix=None, requires_grad=True):
        if not network_prefix:
            raise ValueError('Specify a prefix for a NN. Note that this must be used for all parameters in the desired network.')
        else:
            params_switch_requ_grad = self.getNamedParamList(listInclNames=[network_prefix])
            for i, p in enumerate(params_switch_requ_grad):
                p.requires_grad = requires_grad

    def setRequiresGrad(self, parameter_list=None, requires_grad=True):
        if not parameter_list:
            raise ValueError('Specify a parmeters list. Note that this must be used for all parameters in the desired network.')
        else:
            for i, p in enumerate(parameter_list):
                p.requires_grad = requires_grad

    def reset_parameter_by_name_list(self, name_list):
        # go through every variable name
        for par_name in name_list:
            # check if variables exists
            if hasattr(self, par_name):
                # get variable and set it to var_curr
                var_curr = vars(self)[par_name]
                # reset the value of the parameter
                var_curr.data = var_curr.data / var_curr.data * 1.
            else:
                raise ValueError('Variable {} does not exist.'.format(par_name))

    def create_tot_norm_array(self):
        if self.np_array_tot_grad is None:
            self.np_array_tot_grad = np.array([], dtype='float32')

    def get_gradient_norm(self, iteration=0, path=None, store=False):
        '''
        This function stores the gradient norms and the total norm of all gradients in a dictionary.
        :param iteration: current iteration
        :return: grad_norm_dict_list: dictionary
        '''
        grad_norm_dict_list = list()
        total_norm = 0.
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2).item()
                total_norm += param_norm ** 2
                grad_norm_dict_list.append({'p_name': name, 'norm': param_norm, 'iteration': iteration})
        total_norm = total_norm ** 0.5

        grad_norm_dict_list.append({'p_name': 'total_norm', 'norm': total_norm, 'iteration': iteration})
        if path is None and store:
            np.save('gradient_norm', grad_norm_dict_list)
        elif path is not None and store:
            np.save(os.path.join(path, 'gradient_norm'), grad_norm_dict_list)

        return grad_norm_dict_list

    def write_gradient_norm(self):
        temp_list = list()
        total_norm = 0.
        for name, p in self.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2).item()
                total_norm += param_norm ** 2
                temp_list.append(param_norm)
        total_norm = total_norm ** 0.5
        temp_list.append(total_norm)

        # update numpy array for total norm
        self.create_tot_norm_array()
        self.np_array_tot_grad = np.append(self.np_array_tot_grad, total_norm)

        self.grad_norm_storage_list.append(temp_list)

    def save_gradient_norm_list(self, path):

        if not self.grad_norm_storage_list_names_stored:
            name_list = list()
            for name, p in self.named_parameters():
                if p.grad is not None:
                    name_list.append(name)
            name_list.append('total_norm')
            file_name = os.path.join(path, 'grad_norm_names.txt')
            with open(file_name, 'w') as f:
                f.writelines(['%s ' % item for item in name_list])

        self.grad_norm_storage_list_names_stored = True

        np.savetxt(os.path.join(path, 'grad_norm.txt'), self.grad_norm_storage_list)

    def decode(self, z):
        print('Overwrite this function!')
        quit()

    def getlogpdfold(self, samples, bgpu):

        # for estimating the predictive distribution we need to sample z ~ p(z) and estimate
        # mu and sig^2 for each z

        nz = 4000
        nx = samples.shape[0]

        if bgpu:
            samples_z = torch.randn((nz, self.z_dim)).cuda()
            logpdfsamplesxz = torch.zeros([nx, nz]).cuda()
        else:
            samples_z = torch.randn((nz, self.z_dim))
            logpdfsamplesxz = torch.zeros([nx, nz])

        # return statistics of guassian p(x|z)
        mu, logvar = self.decode(samples_z)

        pxgz = torch.distributions.normal.Normal(loc=mu, scale=logvar.exp().sqrt())
        for i in range(nx):
            logpdfsamplesxz[i, :] = pxgz.log_prob(samples[i, :]).sum(dim=1)

        m, m_pos = logpdfsamplesxz.max(dim=1, keepdim=True)
        xma = logpdfsamplesxz - m
        expxma = torch.exp(xma)
        sumexpxma = expxma.sum(dim=1, keepdim=True)
        logsumtemp = torch.log(sumexpxma)
        logsumtemppm = m + logsumtemp

        logsumtemppm.add_(-np.log(nz))

        return logsumtemppm

    def getlogpdf(self, samples, bgpu, nz=None):

        remember_training = self.training
        self.eval()

        # for estimating the predictive distribution we need to sample z ~ p(z) and estimate
        # mu and sig^2 for each z
        if bgpu and nz is None:
            nz = 1000
        elif nz is None:
            nz = 300

        nx = samples.shape[0]

        if bgpu:
            samples_z = torch.randn((nz, self.z_dim)).cuda()
            logpdfsamplesxz = torch.zeros([nx, nz]).cuda()
        else:
            samples_z = torch.randn((nz, self.z_dim))
            logpdfsamplesxz = torch.zeros([nx, nz])

        # return statistics of guassian p(x|z)
        mu, logvar = self.decode(samples_z)

        pxgz = torch.distributions.normal.Normal(loc=mu, scale=logvar.exp().sqrt())
        for i in range(nx):
            logpdfsamplesxz[i, :] = pxgz.log_prob(samples[i, :]).sum(dim=1)

        m, m_pos = logpdfsamplesxz.max(dim=1, keepdim=True)
        xma = logpdfsamplesxz - m
        expxma = torch.exp(xma)
        sumexpxma = expxma.sum(dim=1, keepdim=True)
        logsumtemp = torch.log(sumexpxma)
        logsumtemppm = m + logsumtemp

        logsumtemppm.add_(-np.log(nz))

        self.training = remember_training

        return logsumtemppm

    def get_log_q_x_given_z(self, samples_x, samples_z):
        with torch.no_grad():
            mu, logvar = self.decode(samples_z.view(-1, self.z_dim))
            pointwiseLogpxgz = -0.5 * F.mse_loss(samples_x, mu, size_average=False, reduce=False)
            weightpxgz = logvar.exp().reciprocal()
            pointwiseWeightedMSElossLogpxgz = pointwiseLogpxgz.mul(weightpxgz)
            logpxgz = pointwiseWeightedMSElossLogpxgz.sum(dim=1)
            logpxgz -= 0.5 * logvar.sum(dim=1)
            logpxgz -= self.x_dim * 0.5 * self.log2pi
        return logpxgz

    def get_log_r_z_given_x(self, samples_x, samples_z):
        with torch.no_grad():
            mu, logvar = self.encode(samples_x)
            pointwiseLogrzgx = -0.5 * F.mse_loss(samples_z, mu, size_average=False, reduce=False)
            weightrzgx = logvar.exp().reciprocal()
            pointwiseWeightedMSElossLogrzgx = pointwiseLogrzgx.mul(weightrzgx)
            logrzgx = pointwiseWeightedMSElossLogrzgx.sum(dim=1)
            logrzgx -= 0.5 * logvar.sum(dim=1)
            logrzgx -= self.z_dim * 0.5 * self.log2pi
        return logrzgx

    def get_log_q_z(self, samples_z):
        with torch.no_grad():
            logpz = self.qz.log_prob(samples_z).sum(dim=1)
        return logpz

    def plotprediction(self, samples, path, postfix='', refsamples=None, refsamples_weights=None):

        settinggaussianmixture = {'setlim': True, 'nbins': 60, 'vmin': 0.1, 'vmax': 20.}
        settingenergyfunctional = {'setlim': True, 'nbins': 40, 'vmin': 0.0001, 'vmax': 1.}

        setting = settingenergyfunctional

        bSetLim = setting['setlim']
        nbins = setting['nbins']
        vmin = setting['vmin']
        vmax = setting['vmax']

        # estimate sample mean and covariance
        mean = samples.mean(axis=0)
        cov = np.cov(samples.T)

        if self.x_dim == 2:
            if refsamples is not None:
                f, ax = plt.subplots(1, 2, sharey=True)#, tight_layout=True)
                # #f.suptitle(r'Prediction: $\mu_1$ = %1.3f  $\mu_2$ = %1.3f $\sigma_1^2$ = %1.3 $\sigma_2^2$ = %1.3' %(mean[0], mean[1], std[0], std[1]) )
                if refsamples_weights is not None:
                    countsref, xedgesref, yedgesref, imref = ax[0].hist2d(refsamples[0, :], refsamples[1, :], weights=refsamples_weights, bins=[nbins, nbins], normed=True, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
                else:
                    countsref, xedgesref, yedgesref, imref = ax[0].hist2d(refsamples[0, :], refsamples[1, :], bins=[nbins, nbins], normed=True, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
                ax[0].set_ylim(-5., 5.)
                ax[0].set_xlim(-4., 4.)
                ax[0].set_ylabel(r'$x_2$')
                ax[0].set_xlabel(r'$x_1$')
                ax[1].set_xlabel(r'$x_1$')
                ylim = ax[0].get_ylim()
                xlim = ax[0].get_xlim()
                #counts, xedges, yedges, im = ax[1].hist2d(samples[:, 0], samples[:, 1], bins=[xedgesref, yedgesref], normed=True, norm=colors.LogNorm(vmin=vmin, vmax=vmax))#vmin=vmin, vmax=vmax))
                counts, xedges, yedges, im = ax[1].hist2d(samples[:, 0], samples[:, 1], bins=[nbins, nbins], normed=True, norm=colors.LogNorm(vmin=vmin, vmax=vmax))#vmin=vmin, vmax=vmax))
                ax[1].set_title('Prediction')
                ax[0].set_title('Target')
                if bSetLim:
                    for axitem in ax:
                        axitem.set_xlim(xlim)
                        axitem.set_ylim(ylim)
                        axitem.grid(ls='dashed')
                        axitem.set_axisbelow(True)
                f.colorbar(imref)
            else:
                f, ax = plt.subplots(1)  # , tight_layout=True)
                #counts, xedges, yedges, im = ax.hist2d(samples[0, :], samples[1, :], bins=[80, 80], normed=True, norm=colors.LogNorm(vmin=1, vmax=200))
                counts, xedges, yedges, im = ax.hist2d(samples[:, 0], samples[:, 1], bins=[nbins, nbins], normed=True, norm=colors.LogNorm())
                if bSetLim:
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1.5, 1.5])
                f.colorbar(im)
            f.savefig(path + '/prediction' + postfix + '.png', bbox_inches='tight')#, pad_inches=0)
            plt.close(f)
        return mean, cov.flatten()

    def plotlatentrep(self, x, z_dim, path, postfix='', iter=-1, x_curr=0, y_curr=0, nprov=False, normaltemp=0, x_train=None, peptide='ala_2', data_dir=None):

        baddactfctannotation = False
        sizedataset = x.shape[0]
        self.eval()
        mu, logvar = self.encode(x)

        munp = mu.data.cpu().numpy()

        ssize = 20
        alpha = 0.1

        # get the color code, markers, and legend addons
        if peptide is 'ala_2':
            from utils_peptide import getcolorcode1527
            colcode, markers, patchlist = getcolorcode1527(ssize=ssize)
        else:
            from utils_peptide import getcolorcodeALA15
            colcode, markers, patchlist, alphaPerSample = getcolorcodeALA15(ramapath=os.path.join(data_dir, 'ala-15'),
                                                                            ssize=ssize, N=sizedataset)
            # load colors, markers and patchlist for ala_15

        if z_dim == 2: #and sizedataset == 1527:

            #fontloc = {'weight': 'normal', 'size': 10}
            #matplotlib.rc('font', **fontloc)

            plt.figure(1)
            f, ax = plt.subplots()

            # this title is just valid if we use no training data different from the test data.
            #if x_train is None:
                #f.suptitle(r'AEVB: Encoded representation of training data: $\boldsymbol{\mu}(\boldsymbol{x}^{(i)})$')

            iA = 29
            iB1 = 932
            iB2 = 566

            # plot N(0,I)
            #n_samples_normal = iA + iB1 + iB2
            n_samples_normal = 4000
            if not nprov:
                normal = np.random.randn(n_samples_normal, 2)
            else:
                normal = normaltemp

            #if x_train is None:
            if False:
                normalpatch = ax.scatter(normal[:, 0], normal[:, 1], c='g', marker='.', s=ssize, alpha=alpha,
                                     label=r'$\boldsymbol{z} \sim \mathcal N (\boldsymbol{0},\boldsymbol{I})$')
                #h,l= ax.get_legend_handles_labels()
                patchlist.append(normalpatch)

            if peptide is 'ala_2':
                x, y = munp[0:iA, 0], munp[0:iA, 1]
                ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize)
                x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
                ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize)
                x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
                ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize)
            else:
                x, y = munp[:, 0], munp[:, 1]
                #[(x * 1.0 / N, 1., 1.) for x in range(N)]
                [ax.scatter(x[i], y[i], c=colcode[i, :], s=10, alpha=alphaPerSample[i]) for i in range(sizedataset)]
                #ax.scatter(x, y, c=colcode, s=10)

            if baddactfctannotation:
                # list of encoder activation functions
                an = []
                an.append(ax.annotate('Encoder activations:', xy=(-2., 2.7), xycoords="data",
                      va="center", ha="center"))
                an.append(ax.annotate(self.listenc[0], xy=(1, 0.5), xycoords=an[0],  # (1,0.5) of the an1's bbox
                                      xytext=(20, 0), textcoords="offset points",
                                      va="center", ha="left",
                                      bbox=dict(boxstyle="round", fc="None")))
                for i in range(1, len(self.listenc)):
                    an.append(ax.annotate(self.listenc[i], xy=(1, 0.5), xycoords=an[i], # (1,0.5) of the an1's bbox
                      xytext=(20, 0), textcoords="offset points",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round", fc="None"),
                      arrowprops=dict(arrowstyle="<-")))
                # va="center", ha="left",

            if x_train is not None:
                #

                # encode the training data
                mu_train, logvar_train = self.encode(x_train)
                munp_train = mu_train.data.cpu().numpy()
                leng_train = munp_train.shape[0]
                # plot the training data
                if iter >= 0:
                    rnd = False
                    a_training_data = 0.6
                    col_training_data = 'C4'
                else:
                    rnd = True
                    a_training_data = 0.7
                    col_training_data = 'y'
                if rnd:
                    train_patch = ax.scatter(munp_train[:, 0]+(np.random.rand(leng_train)-0.5)*0.2, munp_train[:, 1]+(np.random.rand(leng_train)-0.5)*0.2,
                                         c=col_training_data, marker='d', s=ssize*0.9, alpha=a_training_data, label=r'Training Data')
                else:
                    train_patch = ax.scatter(munp_train[:, 0], munp_train[:, 1],
                                         c=col_training_data, marker='d', s=ssize*0.9, alpha=a_training_data, label=r'Training Data')
                patchlist.append(train_patch)

            #ax.set_ylim([-3, 3])
            #ax.set_xlim([-3, 3])
            ax.set_xlabel(r'$z_1$')
            ax.set_ylabel(r'$z_2$')
            ax.grid(ls='dashed')
            ax.set_axisbelow(True)
            #ax.legend(handles=patchlist, loc=1)
            if x_train is None:
                ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=False, shadow=False, ncol=4)
            else:
                ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=False, shadow=False, ncol=3)

            if postfix == '' and iter < 0:
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])

                ticksstep = 1.
                ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                ax.xaxis.set_ticks(ticks)
                ax.yaxis.set_ticks(ticks)

                f.savefig(path+'/lat_rep.pdf', bbox_inches='tight')#, transparent=True)
            elif postfix == '' and iter >= 0:
                ax.scatter(x_curr, y_curr, c='y', marker='*', s=ssize*35)
                #ax.set_ylim([-4, 4])
                ax.set_ylim([-2.5, 2.5])
                #ax.set_xlim([-4, 4])
                ax.set_xlim([-2.5, 2.5])

                #ticksstep = 1.5
                #ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                #ax.xaxis.set_ticks(ticks)
                #ax.yaxis.set_ticks(ticks)

                f.savefig(path + '/lat_rep_vis_' + str(iter) + '.png', bbox_inches='tight')  # , transparent=True)
                return normal
            else:
                ax.set_ylim([-3.5, 3.5])
                ax.set_xlim([-3.5, 3.5])
                f.savefig(path + '/lat_rep' + postfix +'.png', bbox_inches='tight')  # , transparent=True)
            plt.close()
        elif peptide is 'ala_15':

            f, ax = plt.subplots(nrows=z_dim-1, ncols=z_dim-1, sharey=True, sharex=True)

            # this title is just valid if we use no training data different from the test data.
            if x_train is None:
                f.suptitle(r'AEVB: Encoded representation of training data: $\boldsymbol{\mu}(\boldsymbol{x}^{(i)})$')

            iA = 29
            iB1 = 932
            iB2 = 566

            # plot N(0,I)
            #n_samples_normal = iA + iB1 + iB2
            n_samples_normal = 4000
            if not nprov:
                normal = np.random.randn(n_samples_normal, z_dim)
            else:
                normal = normaltemp

            #if x_train is None:
            if False:
                for i in range(z_dim-1):
                    for j in range(i, z_dim-1):
                        if not i == (j + 1):
                            normalpatch = ax[i, j].scatter(normal[:, i], normal[:, j+1], c='g', marker='.', s=ssize, alpha=alpha,
                                         label=r'$\boldsymbol{z} \sim \mathcal N (\boldsymbol{0},\boldsymbol{I})$')
                #h,l= ax.get_legend_handles_labels()
                patchlist.append(normalpatch)

            #TODO IMPLEMENT THIS
            if peptide is 'ala_2':
                x, y = munp[0:iA, 0], munp[0:iA, 1]
                ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize)
                x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
                ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize)
                x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
                ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize)
            else:
                for i in range(z_dim-1):
                    for j in range(i, z_dim-1):
                        if not i == (j + 1):
                            x, y = munp[:, i], munp[:, j+1]
                            #[(x * 1.0 / N, 1., 1.) for x in range(N)]
                            if z_dim > 4:
                                ax[i, j].scatter(x, y, c=colcode, s=10)
                            else:
                                [ax[i, j].scatter(x[l], y[l], c=colcode[l, :], s=10, alpha=alphaPerSample[l]) for l in range(sizedataset)]
                            #ax.scatter(x, y, c=colcode, s=10)
            #TODO IMPLEMENT THIS
            if False and baddactfctannotation:
                # list of encoder activation functions
                an = []
                an.append(ax.annotate('Encoder activations:', xy=(-2., 2.7), xycoords="data",
                      va="center", ha="center"))
                an.append(ax.annotate(self.listenc[0], xy=(1, 0.5), xycoords=an[0],  # (1,0.5) of the an1's bbox
                                      xytext=(20, 0), textcoords="offset points",
                                      va="center", ha="left",
                                      bbox=dict(boxstyle="round", fc="None")))
                for i in range(1, len(self.listenc)):
                    an.append(ax.annotate(self.listenc[i], xy=(1, 0.5), xycoords=an[i], # (1,0.5) of the an1's bbox
                      xytext=(20, 0), textcoords="offset points",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round", fc="None"),
                      arrowprops=dict(arrowstyle="<-")))
                # va="center", ha="left",

            #TODO Implement this if required
            if False and x_train is not None:
                # encode the training data
                mu_train, logvar_train = self.encode(x_train)
                munp_train = mu_train.data.cpu().numpy()
                leng_train = munp_train.shape[0]
                # plot the training data
                for i in range(z_dim-1):
                    for j in range(i, z_dim-1):
                        if not i == (j + 1):
                            train_patch = ax[i, j].scatter(munp_train[:, i]+(np.random.rand(leng_train)-0.5)*0.2, munp_train[:, j+1]+(np.random.rand(leng_train)-0.5)*0.2,
                                                     c='y', marker='d', s=ssize*0.9, alpha=0.7, label=r'Training Data')
                patchlist.append(train_patch)

            #ax.set_ylim([-3, 3])
            #ax.set_xlim([-3, 3])
            for i in range(z_dim - 1):
                for j in range(z_dim - 1):
                    if not i==(j+1):
                        ax[i, j].set_xlabel(r'$z_%d$' % i)
                        ax[i, j].set_ylabel(r'$z_%d$' % j)
                        ax[i, j].set_xlim([-5, 5])
                        ax[i, j].set_ylim([-5, 5])
                        ax[i, j].grid(ls='dashed')

            if False:
                ax.set_axisbelow(True)
                #ax.legend(handles=patchlist, loc=1)
                if x_train is None:
                    ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=False, shadow=False, ncol=4)
                else:
                    ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                              fancybox=False, shadow=False, ncol=3)

            if postfix == '' and iter < 0:
                #ax.set_ylim([-4, 4])
                #ax.set_xlim([-4, 4])
                #ticksstep = 1.
                #ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                #ax.xaxis.set_ticks(ticks)
                #ax.yaxis.set_ticks(ticks)
                f.savefig(path+'/lat_rep.pdf', bbox_inches='tight')#, transparent=True)
            elif postfix == '' and iter >= 0:
                #ax.scatter(x_curr, y_curr, c='y', marker='*', s=ssize*35)
                #ax.set_ylim([-4, 4])
                #ax.set_xlim([-4, 4])
                f.savefig(path + '/lat_rep_vis_' + str(iter) + '.png', bbox_inches='tight')  # , transparent=True)
                return normal
            else:
                #ax.set_ylim([-3.5, 3.5])
                #ax.set_xlim([-3.5, 3.5])
                f.savefig(path + '/lat_rep' + postfix +'.png', bbox_inches='tight')  # , transparent=True)
            plt.close()
        else:
            print('Warining: Representation of data in latent space not possible: z_dim is no 2')

    def plotlatentrepOLD(self, x, z_dim, path, postfix='', iter=-1, x_curr=0, y_curr=0, nprov=False, normaltemp=0):

        baddactfctannotation = False
        sizedataset = x.shape[0]

        if z_dim == 2 and sizedataset == 1527:

            #fontloc = {'weight': 'normal', 'size': 10}
            #matplotlib.rc('font', **fontloc)

            mu, logvar = self.encode(x)

            munp = mu.data.cpu().numpy()

            ssize = 20
            alpha = 0.2

            # get the color code, markers, and legend addons
            from utils_peptide import getcolorcode1527
            colcode, markers, patchlist = getcolorcode1527(ssize=ssize)

            plt.figure(1)
            f, ax = plt.subplots()
            f.suptitle(r'AEVB: Encoded representation of training data: $\boldsymbol{\mu}(\boldsymbol{x}^{(i)})$')

            iA = 29
            iB1 = 932
            iB2 = 566

            # plot N(0,I)
            #n_samples_normal = iA + iB1 + iB2
            n_samples_normal = 4000
            if not nprov:
                normal = np.random.randn(n_samples_normal, 2)
            else:
                normal = normaltemp

            normalpatch = ax.scatter(normal[:, 0], normal[:, 1], c='g', marker='.', s=ssize, alpha=alpha,
                                     label=r'$\boldsymbol{z} \sim \mathcal N (\boldsymbol{0},\boldsymbol{I})$')
            #h,l= ax.get_legend_handles_labels()

            x, y = munp[0:iA, 0], munp[0:iA, 1]
            ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize)
            x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
            ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize)
            x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
            ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize)

            if baddactfctannotation:
                # list of encoder activation functions
                an = []
                an.append(ax.annotate('Encoder activations:', xy=(-2., 2.7), xycoords="data",
                      va="center", ha="center"))
                an.append(ax.annotate(self.listenc[0], xy=(1, 0.5), xycoords=an[0],  # (1,0.5) of the an1's bbox
                                      xytext=(20, 0), textcoords="offset points",
                                      va="center", ha="left",
                                      bbox=dict(boxstyle="round", fc="None")))
                for i in range(1, len(self.listenc)):
                    an.append(ax.annotate(self.listenc[i], xy=(1, 0.5), xycoords=an[i], # (1,0.5) of the an1's bbox
                      xytext=(20, 0), textcoords="offset points",
                      va="center", ha="left",
                      bbox=dict(boxstyle="round", fc="None"),
                      arrowprops=dict(arrowstyle="<-")))
                # va="center", ha="left",

            patchlist.append(normalpatch)

            #ax.set_ylim([-3, 3])
            #ax.set_xlim([-3, 3])
            ax.set_xlabel(r'$z_1$')
            ax.set_ylabel(r'$z_2$')
            #ax.legend(handles=patchlist, loc=1)
            ax.legend(handles=patchlist, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=False, shadow=False, ncol=4)

            if postfix == '' and iter < 0:
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])

                ticksstep = 1.
                ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                ax.xaxis.set_ticks(ticks)
                ax.yaxis.set_ticks(ticks)

                f.savefig(path+'/lat_rep.pdf', bbox_inches='tight')#, transparent=True)
            elif postfix == '' and iter>= 0:
                ax.scatter(x_curr, y_curr, c='y', marker='*', s=ssize*35)
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])
                f.savefig(path + '/lat_rep_vis_' + str(iter) + '.png', bbox_inches='tight')  # , transparent=True)
                return normal
            else:
                ax.set_ylim([-3.5, 3.5])
                ax.set_xlim([-3.5, 3.5])
                f.savefig(path + '/lat_rep' + postfix +'.png', bbox_inches='tight')  # , transparent=True)
            plt.close()
        else:
            print('Warining: Representation of data in latent space not possible: z_dim is no 2')

    def getCov(self):
        raise ValueError('For this model, the called class function is not valid.')

    def getMean(self):
        raise ValueError('For this model, the called class function is not valid.')

    def getMeanNP(self):
        raise ValueError('For this model, the called class function is not valid.')

    def getCovNP(self):
        raise ValueError('For this model, the called class function is not valid.')

class VARmdAngAugGrouped(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmdAngAugGrouped, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        #dim_intermed = 30
        dim_intermed = 100

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = int(self.x_dim / 5)
        self.sizer = int(ncoordtupes * 1)
        self.sizephi = int(ncoordtupes * 2)
        self.sizetheta = int(ncoordtupes * 2)

        self.dec_last_r = nn.Linear(self.x_dim, self.sizer)
        self.dec_last_phi = nn.Linear(self.x_dim, self.sizephi)
        self.dec_last_theta = nn.Linear(self.x_dim, self.sizetheta)

        #self.fcDecLast_logvar = nn.Linear(self.x_dim, self.x_dim)

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc4mu = nn.Linear(dim_intermed, self.z_dim)

        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim) * (-0.), requires_grad=True)
        else:
            self.enc_fc4var = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed, x_dim, bias=True)

        self.dec_fc4 = nn.Linear(dim_intermed, x_dim)

        if self.bfixlogvar:
            #logvarr = torch.nn.Parameter(torch.ones(self.sizer) * (-12.), requires_grad=True)
            #logvarphitheta = torch.nn.Parameter(torch.ones(self.sizephi + self.sizetheta) * (-3.), requires_grad=True)
            #self.dec_logvar = torch.cat((logvarr, logvarphitheta), 0)
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * (-5.), requires_grad=True)
            with torch.no_grad():
                self.dec_logvar[0:21].mul_(2.5)
            #self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * (-5.), requires_grad=True)

        if not hasattr(self, 'dec_logvar'):
            self.dec_fc5 = nn.Linear(x_dim, x_dim)

        ## work with independent variance of predictive model
        #if self.bfixlogvar:
        #    self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim), requires_grad=True)
        #else:
        #    self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
        #    self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #    self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)

    def clampAngular(self, x):
        with torch.no_grad():
            x[:, 0:self.sizer].clamp_(min=0.03, max=0.3)
            x[:, self.sizer:].clamp_(min=-0.999, max=0.999)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.selu(muvar)
        muvar = self.enc_fc3muvar(muvar)
        muvar = F.logsigmoid(muvar)

        if self.bfixenclogvar:
            return self.enc_fc4mu(muvar), self.enc_logvar.repeat(muvar.size(0), 1)
        else:
            return self.enc_fc4mu(muvar), self.enc_fc4var(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        inp = self.dec_fc1(z)
        inp = self.selu(inp)
        inp = self.dec_fc2(inp)
        inp = self.selu(inp)
        inp = self.dec_fc3(inp)
        inp = self.selu(inp)

        #mu = self.tanh(self.dec_fc4(inp))

        #x1 = self.tanh(self.dec_last_r(inp)) + 1.
        x1 = self.sigmoid(self.dec_last_r(inp))
        x1 = x1.mul(0.15)
        x1 = x1.add(0.095)
        x2 = self.tanh(self.dec_last_phi(inp))
        x3 = self.tanh(self.dec_last_theta(inp))

        # assemble all the variables
        mu = torch.cat((x1, x2, x3), 1)

        #mu = inp

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_fc5(inp)

        #if self.bfixlogvar:
        #    batch_size = mu.size(0)
        #    logvar = self.dec_logvar.repeat(batch_size, 1)
        #else:
        #    inpvar = self.tanh(self.dec_fc1var(z))
        #    inpvar = self.tanh(self.dec_fc2var(inpvar))
        #    inpvar = self.dec_fc3var(inpvar)
        #    #inpvar = self.dec_fc4var(inpvar)
        #            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        # to ensure we have proper defined values for r cos/sin of phi and theta
        self.clampAngular(x)

        return x, self.encode(x), mu, logvar

    def forward_vae(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_vae(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize_vae(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


class VARmdDEP(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmdDEP, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        #dim_intermed = 30
        #dim_intermed = 500
        dims = [x_dim, 100, 200, 200, 200, 100, x_dim]

        #TODO check if I can delete enc/dec_list without memory issues
        self.enc_list = []
        for i in range(len(dims)-1):
            self.enc_list.append(nn.Linear(dims[i], dims[i + 1]))
        self.enc_linears = nn.ModuleList(self.enc_list)

        self.dec_list = []
        for i in range(len(dims)-1):
            self.dec_list.append(nn.Linear(dims[len(dims) - i - 1], dims[len(dims) - i - 2]))
        self.dec_linears = nn.ModuleList(self.dec_list)

        # for sigma
        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim) * (-0.5), requires_grad=True)
        else:
            self.enc_linlogvar = nn.Linear(x_dim, self.z_dim)
        self.enc_linmu = nn.Linear(x_dim, self.z_dim)

        self.dec_linfirst = nn.Linear(self.z_dim, x_dim)
        self.dec_linmu = nn.Linear(x_dim, x_dim)
        if self.bfixlogvar:
            #self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * (-2.), requires_grad=True)
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * (-5.), requires_grad=True)

        if not hasattr(self, 'dec_logvar'):
            self.dec_linlogvar = nn.Linear(x_dim, x_dim)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):
        lins = len(self.enc_linears)
        muvar = x
        for i, lin in enumerate(self.enc_linears):
            muvar = lin(muvar)
            if i == lins-1:
                muvar = F.logsigmoid(muvar)
            else:
                muvar = self.selu(muvar)

        if self.bfixenclogvar:
            batch_size = x.shape[0]
            return self.enc_linmu(muvar), self.enc_logvar.repeat(batch_size, 1)
        else:
            return self.enc_linmu(muvar), self.enc_linlogvar(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #print 'reparametrize x'
            #print std
            eps = Variable(std.data.new(std.size()).normal_())
            #with torch.no_grad():
            #    eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        lins = len(self.dec_linears)
        inp = self.selu(self.dec_linfirst(z))
        for i, lin in enumerate(self.dec_linears):
            inp = lin(inp)
            if i == lins-1:
                inp = self.tanh(inp)
            else:
                inp = self.selu(inp)

        mu = self.dec_linmu(inp)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_linlogvar(inp)

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''
        mu, logvar = self.decode(z.view(-1, self.z_dim))
        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def forward_vae(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_vae(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize_vae(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False

class VARmd(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmd, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p, dropout_enc_dec)

        self.use_skip_vae_dec = False
        self.use_skip_vae_enc = False
        #dim_intermed = 30
        #dim_intermed_dec = 120
        #dim_intermed_enc = 170
        dim_intermed_dec = 100
        dim_intermed_enc = 150
        #dim_intermed_dec = 200
        #dim_intermed_enc = 300
        self.dec_VARmixturecomplexvar_bias = 1.e-5
        self.enc_var_bias = 1.e-5
        self.b_add_var_bias = False

        self.dec_drop1 = torch.nn.Dropout(p=dropout_p)
        self.dec_drop2 = torch.nn.Dropout(p=dropout_p)
        self.dec_drop3 = torch.nn.Dropout(p=dropout_p)

        self.enc_drop1 = torch.nn.Dropout(p=dropout_p)
        self.enc_drop2 = torch.nn.Dropout(p=dropout_p)
        self.enc_drop3 = torch.nn.Dropout(p=dropout_p)

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed_enc)
        self.enc_fc2muvar = nn.Linear(dim_intermed_enc, dim_intermed_enc)
        self.enc_fc3muvar = nn.Linear(dim_intermed_enc, dim_intermed_enc, bias=True)
        #self.enc_fc4muvar = nn.Linear(dim_intermed_enc, dim_intermed_enc)
        #self.enc_fc5muvar = nn.Linear(dim_intermed_enc, dim_intermed_enc)
        if self.use_skip_vae_enc:
            self.enc_fc1_skip_f = nn.Linear(dim_intermed_enc, dim_intermed_enc, bias=False)
            self.enc_fc2_skip_f = nn.Linear(dim_intermed_enc, dim_intermed_enc, bias=False)
            self.enc_fc3_skip_f = nn.Linear(dim_intermed_enc, dim_intermed_enc, bias=False)
            self.enc_fc1_skip_x = nn.Linear(x_dim, dim_intermed_enc, bias=False)
            self.enc_fc2_skip_x = nn.Linear(x_dim, dim_intermed_enc, bias=False)
            self.enc_fc3_skip_x = nn.Linear(x_dim, dim_intermed_enc, bias=False)
        #self.enc_fc4muvar = nn.Linear(dim_intermed_enc, dim_intermed_enc)
        #self.enc_fc4muvar = nn.Linear(dim_intermed, dim_intermed)
        #self.enc_fc5muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim) * (-1.), requires_grad=True)
        else:
            self.enc_linlogvar = nn.Linear(dim_intermed_enc, self.z_dim)
        # for mu
        self.enc_linmu = nn.Linear(dim_intermed_enc, self.z_dim)

        # decoder
        # mu and logvar
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed_dec, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=True)
        #self.dec_fc4 = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=True)
        #self.dec_fc5 = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=True)
        #self.dec_fc4 = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=True)
        #self.dec_fc4 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #self.dec_fc5 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #
        # skip-vae
        if self.use_skip_vae_dec:
            self.dec_fc1_skip_f = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=False)
            self.dec_fc2_skip_f = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=False)
            self.dec_fc3_skip_f = nn.Linear(dim_intermed_dec, dim_intermed_dec, bias=False)
            self.dec_fc1_skip_z = nn.Linear(self.z_dim, dim_intermed_dec, bias=False)
            self.dec_fc2_skip_z = nn.Linear(self.z_dim, dim_intermed_dec, bias=False)
            self.dec_fc3_skip_z = nn.Linear(self.z_dim, dim_intermed_dec, bias=False)

        # mu
        self.dec_linmu = nn.Linear(dim_intermed_dec, x_dim)
        # logvar
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * -0.5, requires_grad=True)
        else:
            self.dec_linlogvar = nn.Linear(dim_intermed_dec, x_dim)

        ## work with independent variance of predictive model
        #if self.bfixlogvar:
        #    self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim), requires_grad=True)
        #else:
        #    self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
        #    self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #    self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):

        # muvar = self.enc_fc1muvar(x)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc2muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc3muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc4muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc5muvar(muvar)
        # muvar = self.relu(muvar)

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)

        # use dropout
        if self.dropout_active and 'enc' in self.dropout_enc_dec:
            muvar = self.enc_drop1(muvar)

        #muvar = self.prelu(muvar)
        if self.use_skip_vae_enc:
            muvar = self.relu(self.enc_fc1_skip_f(muvar) + self.enc_fc1_skip_x(x))
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.selu(muvar)

        # use dropout
        if self.dropout_active and 'enc' in self.dropout_enc_dec:
            muvar = self.enc_drop2(muvar)

        if self.use_skip_vae_enc:
            muvar = self.relu(self.enc_fc2_skip_f(muvar) + self.enc_fc2_skip_x(x))

        muvar = self.enc_fc3muvar(muvar)
        #muvar = self.selu(muvar)

        #muvar = self.enc_fc4muvar(muvar)
        #muvar = self.selu(muvar)
        #muvar = self.enc_fc5muvar(muvar)

        #muvar = self.selu(muvar)
        #muvar = self.enc_fc4muvar(muvar)
        #muvar = self.selu(muvar)
        #muvar = self.enc_fc5muvar(muvar)
        #muvar = self.tanh(muvar)
        muvar = F.logsigmoid(muvar)
        #muvar = self.prelu(muvar)
        #muvar = self.celu(muvar)
        if self.use_skip_vae_enc:
            muvar = self.relu(self.enc_fc3_skip_f(muvar) + self.enc_fc3_skip_x(x))


        # muvar = self.enc_fc1muvar(x)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc2muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc3muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc4muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc5muvar(muvar)
        # muvar = F.logsigmoid(muvar)

        if self.bfixenclogvar:
            batch_size = x.shape[0]
            return self.enc_linmu(muvar), self.enc_logvar.repeat(batch_size, 1)
        else:
            return self.enc_linmu(muvar), self.enc_linlogvar(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #print 'reparametrize x'
            #print std
            # TODO delete this and check implementation with the one given below.
            #eps = Variable(std.data.new(std.size()).normal_())
            with torch.no_grad():
                eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # inp = self.dec_fc1(z)
        # inp = self.relu(inp)
        # inp = self.dec_fc2(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc3(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc4(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc5(inp)
        # inp = self.relu(inp)
        # mu = self.dec_linmu(inp)
        inp = self.dec_fc1(z)
        #inp = self.selu(inp)
        inp = self.tanh(inp)

        if self.use_skip_vae_dec:
            inp = self.relu(self.dec_fc1_skip_f(inp) + self.dec_fc1_skip_z(z))

        # use dropout
        if self.dropout_active and 'dec' in self.dropout_enc_dec:
            inp = self.dec_drop1(inp)

        #inp = self.prelu(inp)
        #inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        inp = self.tanh(inp)
        if self.use_skip_vae_dec:
            inp = self.relu(self.dec_fc2_skip_f(inp) + self.dec_fc2_skip_z(z))

        # use dropout
        if self.dropout_active and 'dec' in self.dropout_enc_dec:
            inp = self.dec_drop2(inp)

        #inp = self.tanh(inp)
        inp = self.dec_fc3(inp)
        inp = self.tanh(inp)
        if self.use_skip_vae_dec:
            inp = self.relu(self.dec_fc3_skip_f(inp) + self.dec_fc3_skip_z(z))

        # # use dropout
        # if self.dropout_active:
        #    inp = self.dec_drop3(inp)

        #inp = self.dec_fc4(inp)
        #inp = self.tanh(inp)

        #inp = self.dec_fc5(inp)
        #inp = self.tanh(inp)

        #inp = self.dec_fc4(inp)
        #inp = self.relu(inp)
        #inp = self.dec_fc5(inp)
        #inp = self.relu(inp)
        mu = self.dec_linmu(inp)
        # inp = self.dec_fc1(z)
        # inp = self.tanh(inp)
        # inp = self.dec_fc2(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc3(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc4(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc5(inp)
        # inp = self.tanh(inp)
        # mu = self.dec_linmu(inp)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            if self.b_add_var_bias:
                raise ValueError('Do not use this.')
                logvar = (self.dec_linlogvar(inp).exp() + self.dec_var_bias).log()
            else:
                logvar = self.dec_linlogvar(inp)

        #if self.bfixlogvar:
        #    batch_size = mu.size(0)
        #    logvar = self.dec_logvar.repeat(batch_size, 1)
        #else:
        #    inpvar = self.tanh(self.dec_fc1var(z))
        #    inpvar = self.tanh(self.dec_fc2var(inpvar))
        #    inpvar = self.dec_fc3var(inpvar)
        #    #inpvar = self.dec_fc4var(inpvar)
        #            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        #if self.training or self.bgetlogvar:
        return mu, logvar
        #else:
        #    return mu

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''
        mu, logvar = self.decode(z.view(-1, self.z_dim))
        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def forward_vae(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_vae(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize_vae(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


class VARmdSepDec(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmdSepDec, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        #dim_intermed = 30
        dim_intermed = 100
        self.dec_var_bias = 1.e-5
        self.enc_var_bias = 1.e-5
        self.b_add_var_bias = False

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        #self.enc_fc4muvar = nn.Linear(dim_intermed, dim_intermed)
        #self.enc_fc5muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim) * (-1.), requires_grad=True)
        else:
            self.enc_linlogvar = nn.Linear(dim_intermed, self.z_dim)
        # for mu
        self.enc_linmu = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        # mu and logvar
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #self.dec_fc4 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #self.dec_fc5 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        # mu
        self.dec_linmu = nn.Linear(dim_intermed, x_dim)
        # logvar
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim) * 0., requires_grad=True)
        else:
            self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
            self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc3var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_linlogvar = nn.Linear(dim_intermed, x_dim)

        ## work with independent variance of predictive model
        #if self.bfixlogvar:
        #    self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim), requires_grad=True)
        #else:
        #    self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
        #    self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
        #    self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):

        # muvar = self.enc_fc1muvar(x)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc2muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc3muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc4muvar(muvar)
        # muvar = self.relu(muvar)
        # muvar = self.enc_fc5muvar(muvar)
        # muvar = self.relu(muvar)

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        #muvar = self.prelu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.selu(muvar)
        muvar = self.enc_fc3muvar(muvar)
        #muvar = self.selu(muvar)
        #muvar = self.enc_fc4muvar(muvar)
        #muvar = self.selu(muvar)
        #muvar = self.enc_fc5muvar(muvar)
        muvar = F.logsigmoid(muvar)
        #muvar = self.prelu(muvar)
        #muvar = self.celu(muvar)


        # muvar = self.enc_fc1muvar(x)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc2muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc3muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc4muvar(muvar)
        # muvar = self.selu(muvar)
        # muvar = self.enc_fc5muvar(muvar)
        # muvar = F.logsigmoid(muvar)

        if self.bfixenclogvar:
            batch_size = x.shape[0]
            return self.enc_linmu(muvar), self.enc_logvar.repeat(batch_size, 1)
        else:
            return self.enc_linmu(muvar), self.enc_linlogvar(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            #print 'reparametrize x'
            #print std
            # TODO delete this and check implementation with the one given below.
            #eps = Variable(std.data.new(std.size()).normal_())
            with torch.no_grad():
                eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # inp = self.dec_fc1(z)
        # inp = self.relu(inp)
        # inp = self.dec_fc2(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc3(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc4(inp)
        # inp = self.relu(inp)
        # inp = self.dec_fc5(inp)
        # inp = self.relu(inp)
        # mu = self.dec_linmu(inp)
        inp = self.dec_fc1(z)
        #inp = self.selu(inp)
        inp = self.selu(inp)
        #inp = self.prelu(inp)
        #inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        inp = self.tanh(inp)
        #inp = self.tanh(inp)
        inp = self.dec_fc3(inp)
        inp = self.tanh(inp)
        #inp = self.dec_fc4(inp)
        #inp = self.relu(inp)
        #inp = self.dec_fc5(inp)
        #inp = self.relu(inp)
        mu = self.dec_linmu(inp)
        # inp = self.dec_fc1(z)
        # inp = self.tanh(inp)
        # inp = self.dec_fc2(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc3(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc4(inp)
        # inp = self.tanh(inp)
        # inp = self.dec_fc5(inp)
        # inp = self.tanh(inp)
        # mu = self.dec_linmu(inp)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            if self.b_add_var_bias:
                logvar = (self.dec_linlogvar(inp).exp() + self.dec_var_bias).log()
            else:
                inpvar = self.selu(self.dec_fc1var(z))
                inpvar = self.tanh(self.dec_fc2var(inpvar))
                inpvar = self.tanh(self.dec_fc3var(inpvar))
                logvar = self.dec_linlogvar(inpvar)

        #if self.bfixlogvar:
        #    batch_size = mu.size(0)
        #    logvar = self.dec_logvar.repeat(batch_size, 1)
        #else:
        #    inpvar = self.tanh(self.dec_fc1var(z))
        #    inpvar = self.tanh(self.dec_fc2var(inpvar))
        #    inpvar = self.dec_fc3var(inpvar)
        #    #inpvar = self.dec_fc4var(inpvar)
        #            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''
        mu, logvar = self.decode(z.view(-1, self.z_dim))
        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def forward_vae(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize_vae(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize_vae(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

class VARmixturecomplexDeep(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmixturecomplexDeep, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        #dim_intermed = 30
        dim_intermed = 80

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc4muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc5muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc6muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        self.enc_fc7var = nn.Linear(dim_intermed, self.z_dim)
        self.enc_fc7mu = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc4 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc5 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc6 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc7 = nn.Linear(dim_intermed, self.x_dim, bias=True)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim)*2., requires_grad=True)
        else:
            self.dec_fc5var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc6var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc7var = nn.Linear(dim_intermed,  self.x_dim, bias=True)

        #if not hasattr(self, 'dec_logvar'):
        #    self.dec_fc5 = nn.Linear(self.z_dim, x_dim)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.relu(muvar)
        muvar = self.enc_fc3muvar(muvar)
        muvar = self.relu(muvar)
        muvar = self.enc_fc4muvar(muvar)
        muvar = self.relu(muvar)
        muvar = self.enc_fc5muvar(muvar)
        muvar = self.relu(muvar)
        muvar = self.enc_fc6muvar(muvar)
        muvar = self.relu(muvar)

        return self.enc_fc7mu(muvar), self.enc_fc7var(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        inp = self.dec_fc1(z)
        inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        inp = self.relu(inp)
        inp = self.dec_fc3(inp)
        inp = self.relu(inp)
        inp = self.dec_fc4(inp)
        inp = self.relu(inp)
        inpvar = inp
        inp = self.dec_fc5(inp)
        inp = self.relu(inp)
        inp = self.dec_fc6(inp)
        inp = self.relu(inp)
        inp = self.dec_fc7(inp)

        #mu = self.dec_fc3(inp)
        mu = inp

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:

            inpvar = self.tanh(self.dec_fc5var(inpvar))
            inpvar = self.relu(self.dec_fc6var(inpvar))
            inpvar = self.dec_fc7var(inpvar)

            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


class VARmixturecomplexNotWorking(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmixturecomplex, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        dim_intermed = 30
        #dim_intermed = 50
        #dim_intermed = 100

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim)*1.5, requires_grad=True)
        else:
            self.enc_fc4var = nn.Linear(dim_intermed, self.z_dim)
        self.enc_fc4mu = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc4mu = nn.Linear(dim_intermed, self.x_dim, bias=True)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim)*1.5, requires_grad=True)
        else:
            self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
            self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)
            self.dec_fc4var = nn.Linear(dim_intermed, self.x_dim, bias=True)

        #if not hasattr(self, 'dec_logvar'):
        #    self.dec_fc5 = nn.Linear(self.z_dim, x_dim)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):
        batch_size = x.shape[0]

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.tanh(muvar)
        muvar = self.enc_fc3muvar(muvar)
        muvar = self.tanh(muvar)

        if self.bfixenclogvar:
            return self.enc_fc4mu(muvar), self.enc_logvar.repeat(batch_size, 1)
        else:
            return self.enc_fc4mu(muvar), self.enc_fc4var(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        inp = self.dec_fc1(z)
        inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        inp = self.tanh(inp)
        inp = self.dec_fc3(inp)
        inp = self.tanh(inp)

        #mu = self.dec_fc3(inp)
        mu = self.dec_fc4mu(inp)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            #inpvar = self.tanh(self.dec_fc1var(z))
            #inpvar = self.tanh(self.dec_fc2var(inpvar))
            #inpvar = self.dec_fc3var(inpvar)
            inpvar = self.dec_fc4var(inp)

            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


class VARmixturecomplex(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmixturecomplex, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        #dim_intermed = 30
        dim_intermed = 100

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        if self.bfixenclogvar:
            self.enc_logvar = torch.nn.Parameter(torch.ones(self.z_dim)*1.5, requires_grad=True)
        else:
            self.enc_fc4var = nn.Linear(dim_intermed, self.z_dim)
        self.enc_fc4mu = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, dim_intermed, bias=True)
        self.dec_fc3 = nn.Linear(dim_intermed, self.x_dim, bias=True)
        #self.dec_fc4mu = nn.Linear(dim_intermed, self.x_dim, bias=True)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim)*1.5, requires_grad=True)
        else:
            self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
            self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)
            #self.dec_fc4var = nn.Linear(dim_intermed, self.x_dim, bias=True)

        #if not hasattr(self, 'dec_logvar'):
        #    self.dec_fc5 = nn.Linear(self.z_dim, x_dim)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):
        batch_size = x.shape[0]

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.tanh(muvar)
        muvar = self.enc_fc3muvar(muvar)
        muvar = self.tanh(muvar)

        if self.bfixenclogvar:
            return self.enc_fc4mu(muvar), self.enc_logvar.repeat(batch_size, 1)
        else:
            return self.enc_fc4mu(muvar), self.enc_fc4var(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        inp = self.dec_fc1(z)
        inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        inp = self.tanh(inp)
        inp = self.dec_fc3(inp)
        #inp = self.tanh(inp)

        mu = inp
        #mu = self.dec_fc4mu(inp)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            inpvar = self.tanh(self.dec_fc1var(z))
            inpvar = self.tanh(self.dec_fc2var(inpvar))
            inpvar = self.dec_fc3var(inpvar)
            #inpvar = self.dec_fc4var(inp)

            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False



class VARmixture(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmixture, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        dim_intermed = 2

        # encoder
        # for mu + logvar
        self.enc_fc1muvar = nn.Linear(x_dim, dim_intermed)
        self.enc_fc2muvar = nn.Linear(dim_intermed, dim_intermed)
        #self.enc_fc3muvar = nn.Linear(dim_intermed, dim_intermed)
        # for sigma
        self.enc_fc4var = nn.Linear(dim_intermed, self.z_dim)
        self.enc_fc4mu = nn.Linear(dim_intermed, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, dim_intermed, bias=True)
        self.dec_fc2 = nn.Linear(dim_intermed, self.x_dim, bias=True)
        #self.dec_fc3 = nn.Linear(dim_intermed, self.x_dim, bias=True)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.ones(x_dim)*1.5, requires_grad=True)
        else:
            self.dec_fc1var = nn.Linear(self.z_dim, dim_intermed, bias=True)
            self.dec_fc2var = nn.Linear(dim_intermed, dim_intermed, bias=True)
            self.dec_fc3var = nn.Linear(dim_intermed,  self.x_dim, bias=True)

        #if not hasattr(self, 'dec_logvar'):
        #    self.dec_fc5 = nn.Linear(self.z_dim, x_dim)

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):

        muvar = self.enc_fc1muvar(x)
        muvar = self.selu(muvar)
        muvar = self.enc_fc2muvar(muvar)
        muvar = self.tanh(muvar)
        #muvar = self.enc_fc3muvar(muvar)
        #muvar = self.tanh(muvar)

        return self.enc_fc4mu(muvar), self.enc_fc4var(muvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        inp = self.dec_fc1(z)
        inp = self.tanh(inp)
        inp = self.dec_fc2(inp)
        #inp = self.tanh(inp)
        #inp = self.dec_fc3(inp)
        #inp = self.tanh(inp)

        #mu = self.dec_fc3(inp)
        mu = inp

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            inpvar = self.tanh(self.dec_fc1var(z))
            inpvar = self.tanh(self.dec_fc2var(inpvar))
            inpvar = self.dec_fc3var(inpvar)
            #inpvar = self.dec_fc4var(inpvar)

            logvar = inpvar

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


class VARmod(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar, bfixenclogvar=False, device=torch.device('cpu'), dropout_p=0, dropout_enc_dec=None):
        super(VARmod, self).__init__(args, x_dim, bfixlogvar, bfixenclogvar, device, dropout_p)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.zeros(x_dim), requires_grad=True)

        # encoder
        # for mu
        self.enc_fc10 = nn.Linear(x_dim, self.z_dim)
        # for sigma
        self.enc_fc11 = nn.Linear(x_dim, self.z_dim)

        # decoder
        self.dec_fc1 = nn.Linear(self.z_dim, x_dim, bias=True)

        if not hasattr(self, 'dec_logvar'):
            self.dec_fc5 = nn.Linear(self.z_dim, x_dim)

    # get current covariance matrix
    def getCov(self):
        with torch.no_grad():
            w = self.dec_fc1.weight
            wtw = torch.mm(w, w.transpose(0, 1))
            if hasattr(self, 'dec_logvar'):
                sigsq = self.dec_logvar.exp()
            else:
                raise ValueError('The decoder has not the required format.')
            cov = torch.diag(sigsq) + wtw

        return cov

    def getMean(self):
        with torch.no_grad():
            mu = self.dec_fc1.bias
        return mu

    def getMeanNP(self):
        mu = self.getMean().data
        return mu.cpu().numpy()

    def getCovNP(self):
        cov = self.getCov().data
        return cov.cpu().numpy()

    # in the variational approach, encoding is a mapping from z to x
    def encode(self, x):
        return self.enc_fc10(x), self.enc_fc11(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        mu = self.dec_fc1(z)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_fc5(z)

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, z):
        '''
        forward for VAR is mapping from z to x: p(x|z) reparametrize with x = mu(z; theta) + sigma(z; theta) * epsilon
        with epsilon ~ p(epsilon) = Normal.
        :param z: 'data' or samples from p_theta(z)
        :return: batch[0] mu of q_phi(x|z), batch[1] logvar of q_phi(x|z), mu of p_theta(x|z), logvar of p_theta(x|z)
        '''

        mu, logvar = self.decode(z.view(-1, self.z_dim))

        # x = mu + sigma * epsilon
        x = self.reparameterize(mu, logvar)

        return x, self.encode(x), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


import unittest
import parser, sys

class TestVARJmodel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        print('TestVARJmodel.__init__')
        unittest.TestCase.__init__(self, *args, **kwargs)

        #print('****************')
        #print(sys.argv)
        #print('****************')

        # set up an artificial argument
        self.args = parser.parse_args(['--dataset', 'quad', '--gan_type', 'VARjoint', '--epoch', '30000', '--x_dim',
                                      '2', '--z_dim', '1', '--seed', '3251', '--AEVB', '1', '--samples_pred', '100',
                                      '--samples_per_mean', '1', '--sharedlogvar', '1', '--sharedencoderlogvar', '1',
                                      '--L', '2', '--Z', '2', '--outputfreq', '100', '--freeMemory', '0',
                                      '--batch_size', '100', '--Z', '100', '--stepSched', '200', '--stepSchedresetopt', '1',
                                      '--betaVAE', '1.0', '--separateLearningRate', '0',
                                      '--stepSchedintwidth', '0.05', '--redDescription', '0'])
        print(self.args)

    def setUp(self):
        if not self.args:
            raise ValueError('No arguments has been parsed.')
            quit()
        else:
            from VARJmodel import VARmixturecomplex as VARmod
            self.vaemodel = VARmod(args=self.args, x_dim=2, bfixlogvar=False,
                               bfixenclogvar=False)

    def test_switch(self):

        self.vaemodel.setRequiresGrad(network_prefix='enc_', requires_grad=False)

        paramsenc = self.vaemodel.getNamedParamList(listInclNames=['enc_'])
        enc_requ_grad = [p.requires_grad for p in paramsenc]
        print(enc_requ_grad)

        paramsdec = self.vaemodel.getNamedParamList(listInclNames=['dec_'])
        dec_requ_grad = [p.requires_grad for p in paramsdec]
        print(dec_requ_grad)

        self.vaemodel.setRequiresGrad(network_prefix='enc_', requires_grad=True)

        paramsenc = self.vaemodel.getNamedParamList(listInclNames=['enc_'])
        enc_requ_grad = [p.requires_grad for p in paramsenc]
        print(enc_requ_grad)

        self.assertTrue(True)