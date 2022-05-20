from __future__ import print_function
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# due to plotting purposes
import numpy as np

# join path
import os



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


class VAEparent(nn.Module):
    def __init__(self, args, x_dim, bfixlogvar):
        super(VAEparent, self).__init__()

        self.bplotdecoder = False
        self.bplotencoder = False
        self.bgetlogvar = False

        self.bfixlogvar = bfixlogvar

        self.x_dim = x_dim
        self.z_dim = args.z_dim

        self.listenc = []
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()

    def get_encoding_decoding_variance(self, x):

        mu, logvar = self.encode(x)
        btemp = self.bgetlogvar
        self.bgetlogvar = True
        mu_pred, logvar_pred = self.decode(mu)
        self.bgetlogvar = btemp

        var_decoder = logvar_pred.exp()
        var_encoder = logvar.exp()
        l2norm_var_dec = var_decoder.norm()
        l2norm_var_enc = var_encoder.norm()

        return {'var_encoder': var_encoder, 'var_decoder': var_decoder, 'norm_enc': l2norm_var_enc.data.numpy(), 'norm_dec': l2norm_var_dec.data.numpy()}

    def plotlatentrep(self, x, z_dim, path, postfix='', iter=-1, x_curr=0, y_curr=0, nprov=False, normaltemp=0, x_train=None, peptide='ala_2', data_dir=None):

        baddactfctannotation = False
        sizedataset = x.shape[0]

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
                ax.set_ylim([-4, 4])
                ax.set_xlim([-4, 4])

                ticksstep = 1.
                ticks = np.arange(-4, 4 + ticksstep, step=ticksstep)
                ax.xaxis.set_ticks(ticks)
                ax.yaxis.set_ticks(ticks)

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


class VAEmodangauggroupedsimple(VAEparent):
    def __init__(self, args, x_dim):
        super(VAEmodangauggroupedsimple, self).__init__(args, x_dim)

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = self.x_dim / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2


        # last layer projecting onto data manifold

        self.fcDecLast_r = nn.Linear(self.x_dim, self.sizer)
        self.fcDecLast_phi = nn.Linear(self.x_dim, self.sizephi)
        self.fcDecLast_theta = nn.Linear(self.x_dim, self.sizetheta)

        self.fcDecLast_logvar = nn.Linear(self.x_dim, self.x_dim)

        h1_dim = 20
        h11_dim = 50

        self.fc10 = nn.Linear(x_dim, h11_dim)
        self.fc1 = nn.Linear(h11_dim, h1_dim)
        self.fc21 = nn.Linear(h1_dim, self.z_dim)
        self.fc22 = nn.Linear(h1_dim, self.z_dim)

        self.fc3 = nn.Linear(self.z_dim, h1_dim)
        self.fc31 = nn.Linear(h1_dim, h11_dim)
        self.fc4 = nn.Linear(h11_dim, self.x_dim)


        # create list of layers
        self.fcEnc0 = nn.Linear(self.x_dim, self.z_dim)
        #self.fcEnc = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(nlayers)])
        # last layer projecting onto data manifold
        self.fcEncLast1 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEncLast2 = nn.Linear(self.z_dim, self.z_dim)


    def encode(self, x):
        h10 = self.relu(self.fc10(x))
        h1 = self.relu(self.fc1(h10))
        return self.fc21(h1), self.fc22(h1)


    def decode(self, z):

        h3 = self.relu(self.fc3(z))
        h31 = self.relu(self.fc31(h3))
        x = F.tanh(self.fc4(h31))

        # last layer onto data space
        #x = self.relu(self.fcDecLast(x))

        x1 = F.tanh(self.fcDecLast_r(x)) + 1.
        x2 = F.tanh(self.fcDecLast_phi(x))
        x3 = F.tanh(self.fcDecLast_theta(x))

        # assemble all the variables
        mu = torch.cat((x1, x2, x3), 1)

        logvar = self.fcDecLast_logvar(x)

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



class VAEmodauggrouped(VAEparent):
    def __init__(self, args, x_dim):
        super(VAEmodauggrouped, self).__init__(args, x_dim)


        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = self.x_dim / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2


        # last layer projecting onto data manifold

        self.fcDecLast_r = nn.Linear(self.x_dim, self.sizer)
        self.fcDecLast_phi = nn.Linear(self.x_dim, self.sizephi)
        self.fcDecLast_theta = nn.Linear(self.x_dim, self.sizetheta)

        self.fcDecLast_logvar = nn.Linear(self.x_dim, self.x_dim)

        h1_dim = 100
        h11_dim = 50
        h12_dim = 100
        #h1_dim = 20
        #h11_dim = 50
        #h12_dim = 100
        # encoder
        self.fc10 = nn.Linear(self.x_dim, h12_dim)
        self.fc11 = nn.Linear(h12_dim, h11_dim)
        self.fc21 = nn.Linear(h11_dim, self.z_dim)
        self.fc22 = nn.Linear(h11_dim, self.z_dim)

        # decoder
        self.fc301 = nn.Linear(self.z_dim, self.z_dim)
        self.fc302 = nn.Linear(self.z_dim, self.z_dim)
        self.fc303 = nn.Linear(self.z_dim, self.z_dim)
        self.fc30 = nn.Linear(self.z_dim, h1_dim)
        self.fc31 = nn.Linear(h1_dim, h11_dim)
        self.fc32 = nn.Linear(h11_dim, h12_dim)
        self.fc4 = nn.Linear(h12_dim, self.x_dim)
        self.fc5 = nn.Linear(h12_dim, self.x_dim)

    def encode(self, x):

        self.listenc = ['selu', 'logsig', 'logsig']

        h10 = self.selu(self.fc10(x))
        h11 = self.tanh(self.fc11(h10))
        #h1 = self.tanh(self.fc12(h11))
        #h1 = F.logsigmoid(self.fc12(h11))

        h1np = h11.data.cpu().numpy()
        if (h1np != h1np).any():
            print(self.fc10.parameters())
            print('Error NANA')

        var = self.sigmoid(self.fc22(h11))

        minvar = Variable(torch.ones(1).fill_(1.0e-6))
        var = var + minvar.expand_as(var)
        logvar = var.log()

        return self.fc21(h11), logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            h1np = std.data.cpu().numpy()
            if (h1np < 0.01).any():
                print('Error NANA')
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        x = z

        if self.bplotdecoder:
            xnp = x.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            ssize = 2
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, 6 + 1, sharey='row')
            f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[0].set_title(str(0))

        x01_lin = self.fc301(x)
        x01_relu = self.relu(x01_lin)
        x02_lin = self.fc302(x01_relu)
        x02_relu = self.relu(x02_lin)
        #x03_lin = self.fc303(x02_relu)
        #x03_relu = self.relu(x03_lin)

        # plot output of linear layer
        if self.bplotdecoder:
            xnp = x01_lin.data.numpy()
            icount = 0
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))
            icount = 1
            xnp = x01_relu.data.numpy()
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))
            xnp = x02_lin.data.numpy()
            icount = 2
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))
            icount = 3
            xnp = x02_relu.data.numpy()
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))

            f.savefig('gen_plt.pdf', bbox_inches='tight', transparent=True)

        #h30 = self.selu(self.fc30(z))
        #h31 = self.tanh(self.fc31(h30))
        #h32 = self.tanh(self.fc32(h31))

        h30 = self.selu(self.fc30(z))
        h31 = F.logsigmoid(self.fc31(h30))
        #h32 = self.tanh(self.fc32(h31))



        #h30 = self.sigmoid(self.fc30(z))
        #h31 = self.tanh(self.fc31(h30))
        #h32 = F.logsigmoid(self.fc32(h31))

        #h30 = self.relu(self.fc30(z))
        #h31 = self.relu(self.fc31(h30))
        #h32 = self.relu(self.fc32(h31))

        mut = self.fc4(h31)
        radrange = Variable(torch.ones(1))
        x1 = self.tanh(self.fcDecLast_r(mut)) + radrange.expand_as(self.fcDecLast_r(mut))
        x2 = self.tanh(self.fcDecLast_phi(mut))
        x3 = self.tanh(self.fcDecLast_theta(mut))

        # assemble all the variables
        mu = torch.cat((x1, x2, x3), 1)

        mutnp = mu.data.cpu().numpy()
        if (mutnp != mutnp).any():
            print('Error: nan in mu')

        var = self.sigmoid(self.fc5(h31))
        minvar = Variable(torch.ones(1).fill_(1.0e-6))
        var = var + minvar.expand_as(var)
        logvar = var.log()

        sigsqtem = logvar.exp()
        sigsqnp = sigsqtem.data.cpu().numpy()
        if (sigsqnp != sigsqnp).any():
            print('Error: nan in mu')

        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False



class VAEmod(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar):
        super(VAEmod, self).__init__(args, x_dim, bfixlogvar)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.zeros(x_dim), requires_grad=True)
        #self.logvar = self.dec_logvardiag
        #self.varoffdiag = torch.nn.Parameter(torch.zeros(x_dim*(x_dim-1)/2), requires_grad=True)
        #self.logvar = torch.cat((self.logvardiag, self.varoffdiag), dim=1)

        if False:
            h1_dim = 100
            h11_dim = 300
            h12_dim = 300
        else:
            h1_dim = 50
            h11_dim = 100
            h12_dim = 100

        #h13_dim = 100
        #h1_dim = 20
        #h11_dim = 50
        #h12_dim = 100
        # encoder
        self.enc_fc10 = nn.Linear(x_dim, h12_dim)
        self.enc_fc11 = nn.Linear(h12_dim, h11_dim)
        self.enc_fc12 = nn.Linear(h11_dim, h1_dim)
        self.enc_fc21 = nn.Linear(h1_dim, self.z_dim)
        self.enc_fc22 = nn.Linear(h1_dim, self.z_dim)

        # decoder
        #self.fc301 = nn.Linear(self.z_dim, self.z_dim)
        #self.fc302 = nn.Linear(self.z_dim, self.z_dim)
        #self.fc303 = nn.Linear(self.z_dim, self.z_dim)
        self.dec_fc30 = nn.Linear(self.z_dim, h1_dim)
        self.dec_fc31 = nn.Linear(h1_dim, h11_dim)
        self.dec_fc32 = nn.Linear(h11_dim, h12_dim)
        self.dec_fc4 = nn.Linear(h12_dim, x_dim)

        if not hasattr(self, 'dec_logvar'):
            self.dec_fc5 = nn.Linear(h12_dim, x_dim)

    def encode(self, x):

        #self.listenc = ['selu', 'logsig', 'logsig']
        self.listenc = ['relu', 'relu', 'relu']

        if True:
            h10 = self.selu(self.enc_fc10(x))
            #h11 = self.tanh(self.enc_fc11(h10))
            ##h1 = self.tanh(self.fc12(h11))
            #h1 = self.tanh(self.enc_fc12(h11))
            h11 = self.selu(self.enc_fc11(h10))
            #h1 = self.tanh(self.fc12(h11))
            #h1 = self.selu(self.enc_fc12(h11))
            h1 = F.logsigmoid(self.enc_fc12(h11))
        if False:
            h10 = self.selu(self.enc_fc10(x))
            h11 = self.tanh(self.enc_fc11(h10))
            h1 = self.tanh(self.enc_fc12(h11))
        if False:
            h10 = self.relu(self.enc_fc10(x))
            #h11 = self.tanh(self.enc_fc11(h10))
            ##h1 = self.tanh(self.fc12(h11))
            #h1 = self.tanh(self.enc_fc12(h11))
            h11 = self.relu(self.enc_fc11(h10))
            #h1 = self.tanh(self.fc12(h11))
            h1 = self.relu(self.enc_fc12(h11))

        return self.enc_fc21(h1), self.enc_fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        x = z
        #
        # if self.bplotdecoder:
        #     xnp = x.data.numpy()
        #     nsamples = xnp.shape[0]
        #     # define color range
        #     cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
        #     ssize = 2
        #     # create plot
        #     plt.figure(1)
        #     f, ax = plt.subplots(1, 6 + 1, sharey='row')
        #     f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
        #     ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[0].set_title(str(0))
        #
        # x01_lin = self.fc301(x)
        # x01_relu = self.relu(x01_lin)
        # x02_lin = self.fc302(x01_relu)
        # x02_relu = self.relu(x02_lin)
        # #x03_lin = self.fc303(x02_relu)
        # #x03_relu = self.relu(x03_lin)
        #
        # # plot output of linear layer
        # if self.bplotdecoder:
        #     xnp = x01_lin.data.numpy()
        #     icount = 0
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     icount = 1
        #     xnp = x01_relu.data.numpy()
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     xnp = x02_lin.data.numpy()
        #     icount = 2
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     icount = 3
        #     xnp = x02_relu.data.numpy()
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #
        #     f.savefig('gen_plt.pdf')
        #
        #h30 = self.selu(self.dec_fc30(z))
        #h31 = self.tanh(self.dec_fc31(h30))
        #h32 = self.tanh(self.dec_fc32(h31))

        h30 = self.tanh(self.dec_fc30(z))
        h31 = self.tanh(self.dec_fc31(h30))
        h32 = self.tanh(self.dec_fc32(h31))

        #h30 = self.sigmoid(self.fc30(z))
        #h31 = self.tanh(self.fc31(h30))
        #h32 = F.logsigmoid(self.fc32(h31))

        #h30 = self.relu(self.fc30(z))
        #h31 = self.relu(self.fc31(h30))
        #h32 = self.relu(self.fc32(h31))

        mu = self.dec_fc4(h32)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_fc5(h32)

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False

class VAEmodComplex(VAEparent):
    def __init__(self, args, x_dim, bfixlogvar):
        super(VAEmodComplex, self).__init__(args, x_dim, bfixlogvar)

        # work with independent variance of predictive model
        if self.bfixlogvar:
            self.dec_logvar = torch.nn.Parameter(torch.zeros(x_dim), requires_grad=True)
        #self.logvar = self.dec_logvardiag
        #self.varoffdiag = torch.nn.Parameter(torch.zeros(x_dim*(x_dim-1)/2), requires_grad=True)
        #self.logvar = torch.cat((self.logvardiag, self.varoffdiag), dim=1)

        h1_dim = 50
        h11_dim = 100
        h12_dim = 250
        h13_dim = 500
        #h13_dim = 100
        #h1_dim = 20
        #h11_dim = 50
        #h12_dim = 100
        # encoder
        self.enc_fc10 = nn.Linear(x_dim, h13_dim)
        self.enc_fc11 = nn.Linear(h13_dim, h12_dim)
        self.enc_fc12 = nn.Linear(h12_dim, h11_dim)
        self.enc_fc13 = nn.Linear(h11_dim, h1_dim)
        self.enc_fc21 = nn.Linear(h1_dim, self.z_dim)
        self.enc_fc22 = nn.Linear(h1_dim, self.z_dim)

        # decoder
        #self.fc301 = nn.Linear(self.z_dim, self.z_dim)
        #self.fc302 = nn.Linear(self.z_dim, self.z_dim)
        #self.fc303 = nn.Linear(self.z_dim, self.z_dim)
        self.dec_fc30 = nn.Linear(self.z_dim, h1_dim)
        self.dec_fc31 = nn.Linear(h1_dim, h11_dim)
        self.dec_fc32 = nn.Linear(h11_dim, h12_dim)
        self.dec_fc33 = nn.Linear(h12_dim, h13_dim)
        self.dec_fc4 = nn.Linear(h13_dim, x_dim)

        ## encoder
        #self.enc_fc10 = nn.Linear(x_dim, h13_dim)
        #self.enc_fc11 = nn.Linear(h13_dim, h12_dim)
        #self.enc_fc12 = nn.Linear(h12_dim, h11_dim)
        #self.enc_fc13 = nn.Linear(h11_dim, h1_dim)
        #self.enc_fc21 = nn.Linear(h1_dim, self.z_dim)
        #self.enc_fc22 = nn.Linear(h1_dim, self.z_dim)

        ## decoder
        ##self.fc301 = nn.Linear(self.z_dim, self.z_dim)
        ##self.fc302 = nn.Linear(self.z_dim, self.z_dim)
        ##self.fc303 = nn.Linear(self.z_dim, self.z_dim)
        #self.dec_fc30 = nn.Linear(self.z_dim, h1_dim)
        #self.dec_fc31 = nn.Linear(h1_dim, h11_dim)
        #self.dec_fc32 = nn.Linear(h11_dim, h12_dim)
        #self.dec_fc33 = nn.Linear(h12_dim, h13_dim)
        #self.dec_fc4 = nn.Linear(h13_dim, x_dim)

        if not hasattr(self, 'dec_logvar'):
            self.dec_fc5 = nn.Linear(h13_dim, x_dim)

    def encode(self, x):

        #self.listenc = ['selu', 'logsig', 'logsig']
        self.listenc = ['relu', 'relu', 'relu']

        if True:
            h10 = self.selu(self.enc_fc10(x))
            #h11 = self.tanh(self.enc_fc11(h10))
            ##h1 = self.tanh(self.fc12(h11))
            #h1 = self.tanh(self.enc_fc12(h11))
            h11 = self.selu(self.enc_fc11(h10))
            #h1 = self.tanh(self.fc12(h11))
            #h1 = self.selu(self.enc_fc12(h11))
            h12 = F.logsigmoid(self.enc_fc12(h11))
            h1 = F.logsigmoid(self.enc_fc13(h12))
        if False:
            h10 = self.selu(self.enc_fc10(x))
            h11 = self.tanh(self.enc_fc11(h10))
            h1 = self.tanh(self.enc_fc12(h11))
        if False:
            h10 = self.relu(self.enc_fc10(x))
            #h11 = self.tanh(self.enc_fc11(h10))
            ##h1 = self.tanh(self.fc12(h11))
            #h1 = self.tanh(self.enc_fc12(h11))
            h11 = self.relu(self.enc_fc11(h10))
            #h1 = self.tanh(self.fc12(h11))
            h1 = self.relu(self.enc_fc12(h11))

        return self.enc_fc21(h1), self.enc_fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        x = z
        #
        # if self.bplotdecoder:
        #     xnp = x.data.numpy()
        #     nsamples = xnp.shape[0]
        #     # define color range
        #     cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
        #     ssize = 2
        #     # create plot
        #     plt.figure(1)
        #     f, ax = plt.subplots(1, 6 + 1, sharey='row')
        #     f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
        #     ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[0].set_title(str(0))
        #
        # x01_lin = self.fc301(x)
        # x01_relu = self.relu(x01_lin)
        # x02_lin = self.fc302(x01_relu)
        # x02_relu = self.relu(x02_lin)
        # #x03_lin = self.fc303(x02_relu)
        # #x03_relu = self.relu(x03_lin)
        #
        # # plot output of linear layer
        # if self.bplotdecoder:
        #     xnp = x01_lin.data.numpy()
        #     icount = 0
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     icount = 1
        #     xnp = x01_relu.data.numpy()
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     xnp = x02_lin.data.numpy()
        #     icount = 2
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #     icount = 3
        #     xnp = x02_relu.data.numpy()
        #     ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
        #     ax[icount + 1].set_title(str(icount + 1))
        #
        #     f.savefig('gen_plt.pdf')
        #
        #h30 = self.selu(self.dec_fc30(z))
        #h31 = self.tanh(self.dec_fc31(h30))
        #h32 = self.tanh(self.dec_fc32(h31))

        h30 = self.tanh(self.dec_fc30(z))
        h31 = self.tanh(self.dec_fc31(h30))
        h32 = self.tanh(self.dec_fc32(h31))
        h33 = self.tanh(self.dec_fc33(h32))

        #h30 = self.sigmoid(self.fc30(z))
        #h31 = self.tanh(self.fc31(h30))
        #h32 = F.logsigmoid(self.fc32(h31))

        #h30 = self.relu(self.fc30(z))
        #h31 = self.relu(self.fc31(h30))
        #h32 = self.relu(self.fc32(h31))

        mu = self.dec_fc4(h33)

        if self.bfixlogvar:
            batch_size = mu.size(0)
            logvar = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar = self.dec_fc5(h33)

        #logvar = logvart.expand_as(mu)
        #varsize = logvar.size()
        # test this
        #logvar = Variable(torch.zeros(varsize))

        if self.training or self.bgetlogvar:
            return mu, logvar
        else:
            return mu

        #return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def plotdecoderADAPT(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False

class VAEmodold(nn.Module):
    def __init__(self):
        super(VAEmodold, self).__init__()

        self.x_dim = 784
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

