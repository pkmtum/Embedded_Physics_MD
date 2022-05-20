
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch

# network for peptide dataset
class generator(nn.Module):
    def __initbackup__(self, dim_data=66, dim_latent=2, dim_hidden=(20, 40, 50, 50)):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(dim_latent, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim_hidden[0])
        self.fc3 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc4 = nn.Linear(dim_hidden[1], dim_hidden[2])
        self.fc5 = nn.Linear(dim_hidden[2], dim_hidden[3])
        self.fc6 = nn.Linear(dim_hidden[3], dim_data)
        utils.initialize_weights(self)

    def forwardbackup(self, z):
        # x = F.leaky_relu(self.fc1(z), negative_slope=0.1)
        x = F.relu(self.fc1(z))
        x = F.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return self.fc6(x)

    def __init__(self, dim_data=66, dim_latent=10, dim_hidden=(100, 100, 100)):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(dim_latent, dim_hidden[0])
        #self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        #self.drop2 = torch.nn.Dropout(0.7)
        self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[2])
        self.fc4 = nn.Linear(dim_hidden[2], dim_data)

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = dim_data / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2

        #self.fc5_1 = nn.Linear(self.sizer, self.sizer)
        #self.fc5_2 = nn.Linear(self.sizephi, self.sizephi)
        #self.fc5_3 = nn.Linear(self.sizetheta, self.sizetheta)

        #self.fc4 = nn.Linear(dim_hidden[1], dim_hidden[2])
        #self.fc5 = nn.Linear(dim_hidden[2], dim_hidden[3])
        #self.fc6 = nn.Linear(dim_hidden[3], dim_data)
        utils.initialize_weights(self)

    def forward(self, z):
        x = F.selu(self.fc1(z))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)

        return x

    def __initbbb__(self, dim_data=66, dim_latent=10, dim_hidden=(500, 500)):
        super(generator, self).__init__()

        self.bpredictor = False

        #self.fcinit0 = nn.Linear(dim_latent, 1000)
        #self.fcinit1 = nn.Linear(1000, dim_latent)


        # create list of layers
        self.fc = nn.ModuleList([nn.Linear(dim_latent, dim_latent) for i in range(4)])

        # last layer projecting onto data manifold
        self.fcLast = nn.Linear(dim_latent, dim_data)

        # initialize weights
        utils.initialize_weights(self)


    def plotlatentsamplesbbb(self, n_samples=20, z_dim=2):
        # only visualize if \dim(z) = 2
        if z_dim==2:
            sample_z_ = Variable(torch.rand((n_samples, z_dim)), volatile=True)
            self.bpredictor = True
            samples_x_ = self(sample_z_).data.cpu().numpy()
            self.bpredictor = False


    def forwardbbb(self, z):

        lengnet = len(self.fc)
        x = z

        if self.bpredictor:
            xnp = x.data.numpy()
            nsamples = xnp.shape[0]
            # define color range
            cols = matplotlib.cm.viridis(np.linspace(0, 1, nsamples))
            # create plot
            plt.figure(1)
            f, ax = plt.subplots(1, lengnet + 1, sharey='row')
            f.suptitle(r'Generator $G(z;\theta_g)$: Layer')
            ax[0].scatter(xnp[:, 0], xnp[:, 1], color=cols)
            ax[0].set_title(str(0))

        # high dimensional step in beteween
        #x = F.tanh(self.fcinit0(x))
        #x = F.tanh(self.fcinit1(x))

        icount = 0
        for fcx in self.fc:

            # linear layer
            x = fcx(x)

            # plot output of linear layer
            if self.bpredictor:
                xnp = x.data.numpy()
                ax[icount + 1].scatter(xnp[:, 0], xnp[:, 1], color=cols)
                ax[icount + 1].set_title(str(icount + 1))

            # non-linear activation function
            if icount % 2 == 1:
                #x = F.leaky_relu(fcx(x))
                x = F.tanh(x)
            else:
                x = F.tanh(x)

            icount = icount + 1

        if self.bpredictor:
            f.savefig('gen_plt.pdf')

        # last layer onto data space
        x = self.fcLast(x)

        return x

class discriminator(nn.Module):
    def __initbackup__(self, dim_data=66, dim_hidden=(50, 50, 40, 30)):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_data, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[2])
        self.fc4 = nn.Linear(dim_hidden[2], dim_hidden[3])
        self.fc5 = nn.Linear(dim_hidden[3], 1)
        utils.initialize_weights(self)

    def forwardbackup(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.tanh(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return F.sigmoid(self.fc5(x))

    def __initOld__(self, dim_data=66, dim_hidden=(50, 50, 40, 30)):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_data, dim_data)
        self.fc2 = nn.Linear(dim_data, dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[2])
        self.fc4 = nn.Linear(dim_hidden[2], dim_hidden[3])
        self.fc5 = nn.Linear(dim_hidden[3], 1)
        utils.initialize_weights(self)

    def forwardOld(self, x):
        x = F.softplus(self.fc1(x), beta=0.08)
        x = F.tanh(self.fc2(x))
        #x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.tanh(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return F.sigmoid(self.fc5(x))

    def __init__(self, dim_data=66, dim_hidden=(100, 100, 100), boutsig=False):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_data, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], 1)
        utils.initialize_weights(self)

        self.outsig = boutsig

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        if self.outsig:
            x = F.sigmoid(x)

        return x
