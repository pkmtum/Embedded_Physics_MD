import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# due to plotting purposes
import numpy as np



import matplotlib

matplotlib.use('Agg')

font = {'weight' : 'normal',
        'size'   : 5}

#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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
    def __init__(self, args, x_dim):
        super(VAEparent, self).__init__()

        self.bplotdecoder = False
        self.bplotencoder = False
        self.bgetlogvar = False

        self.x_dim = x_dim
        self.z_dim = args.z_dim

        self.listenc = []

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()

    def plotlatentrep(self, x, z_dim):

        sizedataset = x.shape[0]

        if z_dim == 2 and sizedataset == 1527:

            fontloc = {'weight': 'normal',
                    'size': 10}

            matplotlib.rc('font', **fontloc)

            mu, logvar = self.encode(x)

            munp = mu.data.cpu().numpy()

            from utils_peptide import getcolorcode1527
            colcode, markers, patchlist = getcolorcode1527()

            # plot N(0,I)
            normal = np.random.randn(1000, 2)

            ssize = 5
            alpha = 0.2
            alphadata = 0.3
            plt.figure(1)
            f, ax = plt.subplots()
            f.suptitle(r'Encoded representation of training data: $\boldsymbol{\mu}(\boldsymbol{x}^{(i)})$')

            normalpatch = ax.scatter(normal[:, 0], normal[:, 1], c='g', s=ssize, alpha=alpha, label=r'$z \sim \mathcal N (0,1)$')
            #h,l= ax.get_legend_handles_labels()

            iA = 29
            iB1 = 932
            iB2 = 566

            x, y = munp[0:iA, 0], munp[0:iA, 1]
            ax.scatter(x, y, c=colcode[0:iA], marker=markers[0], s=ssize, alpha=alphadata)
            x, y = munp[iA:iA + iB1, 0], munp[iA:iA + iB1, 1]
            ax.scatter(x, y, c=colcode[iA:iA+iB1], marker=markers[1], s=ssize, alpha=alphadata)
            x, y = munp[iA + iB1:iA + iB1 + iB2, 0], munp[iA + iB1:iA + iB1 + iB2, 1]
            ax.scatter(x, y, c=colcode[iA+iB1:iA+iB1+iB2], marker=markers[2], s=ssize, alpha=alphadata)

            # list of encoder activation functions

            if bool(self.listenc):
                an = []
                an.append(ax.annotate('Encoder Activations:', xy=(-2., 2.7), xycoords="data",
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

            ax.set_ylim([-3,3])
            ax.set_xlim([-3, 3])
            ax.set_xlabel('z_1')
            ax.set_ylabel('z_2')
            ax.legend(handles=patchlist, loc=1)
            f.savefig('lat_rep.pdf', bbox_inches='tight')
        else:
            print 'Warining: Representation of data in latent space not possible: z_dim is no 2'


class VAEmodangauggroupedlong(nn.Module):
    def __init__(self, args, x_dim):
        super(VAEmodangauggroupedlong, self).__init__()

        self.bplotdecoder = False
        self.bplotencoder = False

        self.x_dim = x_dim
        self.z_dim = args.z_dim

        nlayers = 6

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = self.x_dim / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2

        # create list of layers
        self.fcDec = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(nlayers)])

        # last layer projecting onto data manifold

        self.fcDecLast = nn.Linear(self.z_dim, self.x_dim)

        self.fcDecLast_r = nn.Linear(self.x_dim, self.sizer)
        self.fcDecLast_phi = nn.Linear(self.x_dim, self.sizephi)
        self.fcDecLast_theta = nn.Linear(self.x_dim, self.sizetheta)

        h1_dim = 15
        h11_dim = 35

        self.fc1 = nn.Linear(x_dim, h1_dim)
        self.fcShowIn = nn.Linear(h1_dim, self.z_dim)
        self.fcShowOut = nn.Linear(self.z_dim, h1_dim)
        self.fc21 = nn.Linear(h1_dim, self.z_dim)
        self.fc22 = nn.Linear(h1_dim, self.z_dim)

        self.fc3 = nn.Linear(self.z_dim, h1_dim)
        self.fc31 = nn.Linear(h1_dim, h11_dim)
        self.fc4 = nn.Linear(h11_dim, self.x_dim)


        # create list of layers
        self.fcEnc0 = nn.Linear(self.x_dim, self.z_dim)
        self.fcEnc = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(nlayers)])
        # last layer projecting onto data manifold
        self.fcEncLast1 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEncLast2 = nn.Linear(self.z_dim, self.z_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))

        hShowIn = self.fcShowIn(h1)
        hShowInAct = F.tanh(hShowIn)
        hShowOut = self.fcShowOut(hShowInAct)
        hShowOutAct = self.relu(hShowOut)

        if self.bplotencoder:
            xnp = hShowIn.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, 1 + 1)#, sharey='row')
            f.suptitle(r'Encoder $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols)
            ax[0].set_title(str(0))
            xnp = hShowInAct.data.numpy()
            ax[1].scatter(xnp[:, 0], xnp[:, 1], color=cols)
            ax[1].set_title(str(1))

            return f


        return self.fc21(hShowOutAct), self.fc22(hShowOutAct)


    def decode(self, z):

        h3 = self.relu(self.fc3(z))
        h31 = self.relu(self.fc31(h3))
        x = self.relu(self.fc4(h31))

        # last layer onto data space
        #x = self.relu(self.fcDecLast(x))

        x1 = F.tanh(self.fcDecLast_r(x)) + 1.
        x2 = F.tanh(self.fcDecLast_phi(x))
        x3 = F.tanh(self.fcDecLast_theta(x))

        # assemble all the variables
        x = torch.cat((x1, x2, x3), 1)

        return x

    def encodeold(self, x):

        lengnet = len(self.fcEnc)

        x = self.fcEnc0(x)

        if self.bplotencoder:
            xnp = x.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, lengnet + 1)#, sharey='row')
            f.suptitle(r'Encoder $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols)
            ax[0].set_title(str(0))

        x = F.relu(x)

        icount = 0
        for fcx in self.fcEnc:

            # linear layer
            x = fcx(x)

            # plot output of linear layer
            if self.bplotencoder:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols)
                ax[icount + 1].set_title(str(icount + 1))

            # non-linear activation function
            if icount % 2 == 1:
                #x = F.leaky_relu(fcx(x))
                x = F.relu(x)
            else:
                x = F.tanh(x)

            icount = icount + 1

        if self.bplotencoder:
            return f
            #f.savefig('enc_plt.pdf')

        # last layer onto data space
        xinp = self.relu(x)
        z1 = self.fcEncLast1(xinp)
        z2 = self.fcEncLast2(xinp)

        return z1, z2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decodeold(self, z):

        lengnet = len(self.fcDec)
        ssize = 2
        x = z

        if self.bplotdecoder:
            xnp = x.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = colorpointsgaussian(xnp, nsamples=nsamples, name_colmap='viridis')
            #cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, lengnet + 1)#, sharey='row')
            f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[0].set_title(str(0))

        icount = 0
        for fcx in self.fcDec:

            # linear layer
            x = fcx(x)

            # plot output of linear layer
            if self.bplotdecoder:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[icount + 1].set_title(str(icount + 1))

            # non-linear activation function
            x = self.relu(x)
            #if icount % 2 == 1:
                #x = F.leaky_relu(fcx(x))
             #   x = self.relu(x)
            #else:
            #    x = F.tanh(x)

            icount = icount + 1

        if self.bplotdecoder:
            f.savefig('gen_plt.pdf')

        # last layer onto data space
        x = self.relu(self.fcDecLast(x))

        x1 = F.tanh(self.fcDecLast_r(x[:, :self.sizer])) + 1.
        x2 = F.tanh(self.fcDecLast_phi(x[:, self.sizer:(self.sizer + self.sizephi)]))
        x3 = F.tanh(self.fcDecLast_theta(x[:, (self.sizer + self.sizephi):(self.sizer + self.sizephi + self.sizetheta)]))

        # assemble all the variables
        x = torch.cat((x1, x2, x3), 1)

        return x

    def plotdecoderold(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False

    def plotencoder(self, x, z_dim=2, strindex=''):
        # only visualize if \dim(z) = 2
        if z_dim == 2:
            self.bplotencoder = True
            fig = self.encode(x)
            fig.savefig('plt_enc_' + strindex + '.png')
            self.bplotencoder = False

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEmodangauggroupedsimple(VAEparent):
    def __init__(self, args, x_dim):
        super(VAEmodangauggroupedsimple, self).__init__(args, x_dim)

        self.bplotdecoder = False
        self.bplotencoder = False

        self.x_dim = x_dim
        self.z_dim = args.z_dim

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = self.x_dim / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2


        # last layer projecting onto data manifold
        self.fcDecLast = nn.Linear(self.z_dim, self.x_dim)

        self.fcDecLast_r = nn.Linear(self.x_dim, self.sizer)
        self.fcDecLast_phi = nn.Linear(self.x_dim, self.sizephi)
        self.fcDecLast_theta = nn.Linear(self.x_dim, self.sizetheta)

        h1_dim = 100
        h11_dim = 100

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

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h10 = self.selu(self.fc10(x))
        h1 = self.tanh(self.fc1(h10))
        return self.fc21(h1), self.fc22(h1)


    def decode(self, z):

        h3 = self.selu(self.fc3(z))
        h31 = self.tanh(self.fc31(h3))
        x = self.tanh(self.fc4(h31))

        # last layer onto data space
        #x = self.relu(self.fcDecLast(x))

        x1 = self.tanh(self.fcDecLast_r(x)) + 1.
        x2 = self.tanh(self.fcDecLast_phi(x))
        x3 = self.tanh(self.fcDecLast_theta(x))

        # assemble all the variables
        x = torch.cat((x1, x2, x3), 1)

        return x

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

class VAEmodcoordlong(VAEparent):
    def __init__(self, args, x_dim):
        super(VAEmodcoordlong, self).__init__(args, x_dim)

        self.bplotdecoder = False
        self.bplotencoder = False

        nlayers = 2


        # create list of layers
        self.fcDec = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(nlayers)])
        # last layer projecting onto data manifold

        self.fcDec0 = nn.Linear(self.z_dim, self.z_dim)
        self.fcDec1 = nn.Linear(self.z_dim, 100)
        self.fcDec2 = nn.Linear(100, 100)
        self.fcDec3 = nn.Linear(100, self.x_dim)

        self.fcDecLast = nn.Linear(self.x_dim, self.x_dim)

        # create list of layers
        self.fcEnc0 = nn.Linear(self.x_dim, 200)
        self.fcEnc00 = nn.Linear(200, self.z_dim)
        self.fcEnc1 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEnc2 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEnc3 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEnc4 = nn.Linear(self.z_dim, self.z_dim)


        self.fcEnc = nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(nlayers)])
        # last layer projecting onto data manifold
        self.fcEncLast1 = nn.Linear(self.z_dim, self.z_dim)
        self.fcEncLast2 = nn.Linear(self.z_dim, self.z_dim)

    def encode(self, x):

        lengnet = len(self.fcEnc)
        ssize = 2

        x = self.fcEnc0(x)
        x = self.selu(x)
        x = self.fcEnc00(x)

        if self.bplotencoder:
            xnp = x.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, lengnet + 1)#, sharey='row')
            f.suptitle(r'Encoder $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[0].set_title(str(0))

        x = self.tanh(x)


        x = self.fcEnc1(x)

        icount = 0
        # plot output of linear layer
        if self.bplotencoder:
            xnp = x.data.numpy()
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))

        x = self.tanh(x)

        x = self.fcEnc2(x)

        icount = 1
        # plot output of linear layer
        if self.bplotencoder:
            xnp = x.data.numpy()
            ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
            ax[icount + 1].set_title(str(icount + 1))

        x = self.tanh(x)

        x = self.tanh(self.fcEnc3(x))
        x = self.tanh(self.fcEnc4(x))

        if self.bplotencoder:
            return f
            #f.savefig('enc_plt.pdf')




        '''
        
        icount = 0
        for fcx in self.fcEnc:

            # linear layer
            x = fcx(x)

            # plot output of linear layer
            if self.bplotencoder:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[icount + 1].set_title(str(icount + 1))

            # non-linear activation function
            #if icount % 2 == 1:
            #    #x = F.leaky_relu(fcx(x))
            #    x = self.tanh(x)
            #else:
            #    x = F.logsigmoid(x)
            x = F.logsigmoid(x)

            icount = icount + 1

        if self.bplotencoder:
            return f
            #f.savefig('enc_plt.pdf')
        
        '''

        # last layer onto data space
        #xinp = F.logsigmoid(x)
        z1 = self.fcEncLast1(x)
        z2 = self.fcEncLast2(x)

        return z1, z2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):


        x = self.tanh(self.fcDec3(self.tanh(self.fcDec2(self.fcDec1(self.selu(self.fcDec0(z)))))))
        if False:
            lengnet = len(self.fcDec)
            ssize = 2
            x = z

            if self.bplotdecoder:
                xnp = x.data.numpy()
                nsamples = xnp.shape[0]
                # define color range
                cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
                # create plot
                plt.figure(1)
                f, ax = plt.subplots(1, lengnet + 1, sharey='row')
                f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
                ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[0].set_title(str(0))

            x = self.fcDec0(x)

            icount = 0
            if self.bplotdecoder:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[icount + 1].set_title(str(icount + 1))


            x = self.selu(x)

            x = self.fcDec1(x)

            icount = 1
            if self.bplotdecoder:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[icount + 1].set_title(str(icount + 1))

            x = self.tanh(x)

            x = self.tanh(self.fcDec2(x))

            '''
            if self.bplotdecoder:
                xnp = x.data.numpy()
                nsamples = xnp.shape[0]
                # define color range
                cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
                # create plot
                plt.figure(1)
                f, ax = plt.subplots(1, lengnet + 1, sharey='row')
                f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
                ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                ax[0].set_title(str(0))
    
            icount = 0
            for fcx in self.fcDec:
    
                # linear layer
                x = fcx(x)
    
                # plot output of linear layer
                if self.bplotdecoder:
                    xnp = x.data.numpy()
                    ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols, s=ssize)
                    ax[icount + 1].set_title(str(icount + 1))
    
                # non-linear activation function
                #if icount == 0:
                #    x = self.selu(x)
                #else:
                #    x = self.tanh(x)
                x = self.tanh(x)
    
    #            if icount % 2 == 1:
    #                #x = F.leaky_relu(fcx(x))
    #                x = F.leaky_relu(x)
    #           else:
    #                x = F.tanh(x)
    
                icount = icount + 1
    
            if self.bplotdecoder:
                f.savefig('gen_plt.png')
            '''

            x = self.tanh(self.fcDec3(x))
            # last layer onto data space
        x = self.fcDecLast(x)

        return x

    def plotdecoder(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False

    def plotencoder(self, x, z_dim=2, strindex=''):
        # only visualize if \dim(z) = 2
        if z_dim == 2:
            self.bplotencoder = True
            fig = self.encode(x)
            fig.savefig('plt_enc_' + strindex + '.png')
            self.bplotencoder = False

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEmod(nn.Module):
    def __init__(self, args, x_dim):
        super(VAEmod, self).__init__()

        self.x_dim = x_dim
        h1_dim = 15
        h11_dim = 35
        z_dim = args.z_dim

        self.bplotdecoder = False

        # encoder
        self.fc1 = nn.Linear(x_dim, h1_dim)
        self.fc21 = nn.Linear(h1_dim, z_dim)
        self.fc22 = nn.Linear(h1_dim, z_dim)

        # decoder
        self.fc301 = nn.Linear(z_dim, z_dim)
        self.fc302 = nn.Linear(z_dim, z_dim)
        self.fc303 = nn.Linear(z_dim, z_dim)
        self.fc31 = nn.Linear(z_dim, h1_dim)
        self.fc32 = nn.Linear(h1_dim, h11_dim)
        self.fc4 = nn.Linear(h11_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
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

            f.savefig('gen_plt.pdf')

        h31 = self.relu(self.fc31(x02_relu))
        h32 = self.relu(self.fc32(h31))
        return self.fc4(h32)
        #return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def plotdecoder(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.randn((n_samples, z_dim)), volatile=True)
            self.bplotdecoder = True
            samples_x_ = self.decode(sample_z_).data.cpu().numpy()
            self.bplotdecoder = False


class VAEmodsimple(VAEparent):
    def __init__(self, args, x_dim):
        super(VAEmodsimple, self).__init__(args, x_dim)

        h1_dim = 100
        h11_dim = 100
        h12_dim = 100

        # encoder

        self.fc10 = nn.Linear(x_dim, h12_dim)
        self.fc11 = nn.Linear(h12_dim, h11_dim)
        self.fc12 = nn.Linear(h11_dim, h1_dim)
        self.fc21 = nn.Linear(h1_dim, self.z_dim)
        self.fc22 = nn.Linear(h1_dim, self.z_dim)

        # decoder
        self.fc30 = nn.Linear(self.z_dim, h1_dim)
        self.fc31 = nn.Linear(h1_dim, h11_dim)
        self.fc32 = nn.Linear(h11_dim, h12_dim)
        self.fc4 = nn.Linear(h12_dim, x_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):

        self.listenc = ['selu', 'logsig', 'logsig']

        h10 = self.selu(self.fc10(x))
        h11 = F.logsigmoid(self.fc11(h10))
        h1 = F.logsigmoid(self.fc12(h11))
        #h1 = self.selu(self.fc12(h11))

        #h10 = self.sigmoid(self.fc1(x))
        #h11 = self.sigmoid(self.fc11(h10))
        #h1 = self.sigmoid(self.fc12(h11))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        h30 = self.selu(self.fc30(z))
        h31 = self.tanh(self.fc31(h30))
        h32 = self.tanh(self.fc32(h31))

        #h3 = self.tanh(self.fc3(z))

        return self.fc4(h32)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEmodold(nn.Module):
    def __init__(self):
        super(VAEmodold, self).__init__()

        self.x_dim = 784
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

