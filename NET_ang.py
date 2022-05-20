
import utils
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, dim_data=66, dim_latent=10, dim_hidden=(100,)):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(dim_latent, 100)
        self.fc2 = nn.Linear(100, dim_hidden[0])
        self.fc3 = nn.Linear(dim_hidden[0], dim_data)
        #self.fc4 = nn.Linear(dim_hidden[1], dim_hidden[2])
        #self.fc5 = nn.Linear(dim_hidden[2], dim_hidden[3])
        #self.fc6 = nn.Linear(dim_hidden[3], dim_data)
        utils.initialize_weights(self)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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

    def __init__(self, dim_data=66, dim_hidden=(50, 50, 40, 30), boutsig=False):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(dim_data, dim_data)
        self.fc2 = nn.Linear(dim_data, 100)
        self.fc3 = nn.Linear(100, 1)
        utils.initialize_weights(self)

        self.outsig = boutsig

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.fc3(x)

        # in case we need sigmoid output
        if self.outsig:
            x = F.sigmoid(x)

        return x
