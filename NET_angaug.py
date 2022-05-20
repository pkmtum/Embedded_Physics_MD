
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch

# network for peptide dataset
class generator(nn.Module):
    def __initbackup__(self, dim_data=66, dim_latent=2, dim_hidden=(500, 500, 100, 50)):
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

    def __init__(self, dim_data=66, dim_latent=10, dim_hidden=(500, 100)):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(dim_latent, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim_hidden[0])
        self.drop1 = nn.Dropout(p=0.7)
        self.fc3 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.drop2 = nn.Dropout(p=0.7)
        self.fc4 = nn.Linear(dim_hidden[1], dim_data)

        # separate last layer in (r, sin \phi cos \phi, sin \theta cos \theta)
        # size of each group:
        ncoordtupes = dim_data / 5
        self.sizer = ncoordtupes * 1
        self.sizephi = ncoordtupes * 2
        self.sizetheta = ncoordtupes * 2

        self.fc5_1 = nn.Linear(dim_data, self.sizer)
        self.fc5_2 = nn.Linear(dim_data, self.sizephi)
        self.fc5_3 = nn.Linear(dim_data, self.sizetheta)

        #self.fc4 = nn.Linear(dim_hidden[1], dim_hidden[2])
        #self.fc5 = nn.Linear(dim_hidden[2], dim_hidden[3])
        #self.fc6 = nn.Linear(dim_hidden[3], dim_data)
        utils.initialize_weights(self)

    def forward(self, z):
        x = F.selu(self.fc1(z))
        #x = F.tanh(self.drop1(self.fc2(x)))
        #x = F.tanh(self.drop2(self.fc3(x)))
        x = F.tanh(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.tanh(self.fc4(x))

        x1 = F.tanh(self.fc5_1(x)) + 1. # F.tanh(self.fc5_1(x[:, :self.sizer])) + 1.
        x2 = F.tanh(self.fc5_2(x)) #  F.tanh(self.fc5_2(x[:, self.sizer:(self.sizer + self.sizephi)]))
        x3 = F.tanh(self.fc5_3(x)) # F.tanh(self.fc5_3(x[:, (self.sizer + self.sizephi):(self.sizer + self.sizephi + self.sizetheta)]))

        # assemble all the variables
        x = torch.cat((x1, x2, x3), 1)

        return x

        # size of coordinate representation
        #for i in range(0,)
        #x[:, :21] = F.relu(x[:, :21])
        #x[:, 21:] = F.tanh(x[:, 21:])

        # go through all variables
        #x[:, 0:sx[1]:sizecoordtuple] = F.relu(x[:, 0:sx[1]:sizecoordtuple])
        #x[:, 1:sx[1]:sizecoordtuple] = F.tanh(x[:, 1:sx[1]:sizecoordtuple])
        #x[:, 2:sx[1]:sizecoordtuple] = F.tanh(x[:, 2:sx[1]:sizecoordtuple])
        #x[:, 3:sx[1]:sizecoordtuple] = F.tanh(x[:, 3:sx[1]:sizecoordtuple])
        #x[:, 4:sx[1]:sizecoordtuple] = F.tanh(x[:, 4:sx[1]:sizecoordtuple])

        #for i in range(0, sx[1]):
        #    if i % sizecoordtuple == 0:
        #        x[:, i] = F.relu(x[:, i])
        #    elif i % sizecoordtuple == 1:
        #        x[:, i] = F.tanh(x[:, i])
        #    elif i % sizecoordtuple == 2:
        #        x[:, i] = F.tanh(x[:, i])
        #    elif i % sizecoordtuple == 3:
        #        x[:, i] = F.tanh(x[:, i])
        #    elif i % sizecoordtuple == 4:
        #        x[:, i] = F.tanh(x[:, i])
        #    else:
        #        print('Warining: Wrong dimension in coordinates in last layer of generator model.')

        #x[:, :self.sizer] = F.tanh(x[:, :self.sizer]) + 1
        #x[:, self.sizer:] = F.tanh(x[:, self.sizer:])
        #return x


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

    def __init__(self, dim_data=66, dim_hidden=(500, 500, 40, 30), boutsig=False):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_data, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc3 = nn.Linear(dim_hidden[1], 1)
        utils.initialize_weights(self)

        self.outsig = boutsig

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        if self.outsig:
            x = F.sigmoid(x)

        return x
