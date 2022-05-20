from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from torch.utils.data import DataLoader

from VAEmodel import VAEmod


class TensorDatasetDataOnly(torch.utils.data.Dataset):
    """Dataset wrapping only data tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
    """

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
#


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--z_dim', type=int, default=20,
                    help='Dimension of the latent space.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if bool(args.gpu_mode):
    torch.cuda.manual_seed(args.seed)


angulardata = 'no'
bClusterND = False
dataset = 'a_1000'
batch_size = args.batch_size

# is using angular data set, add postfix of the data
if angulardata == 'ang':
    angpostfix = '_ang'
elif angulardata == 'ang_augmented':
    angpostfix = '_ang_augmented'
elif angulardata == 'ang_auggrouped':
    angpostfix = '_ang_auggrouped'
else:
    angpostfix = ''

# load dataset
if bClusterND:
    data_dir = '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide'
else:
    data_dir = '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide'
# data_dir = 'data/peptide'
if dataset == 'm_1527':
    # 1527 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_mixed_1527' + angpostfix + '.txt').T)
elif dataset == 'samples':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_samples' + angpostfix + '.txt').T)
elif dataset == 'm_526':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_mixed_526' + angpostfix + '.txt').T)
elif dataset == 'm_10437':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_mixed_10537' + angpostfix + '.txt').T)
elif dataset == 'a_1000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_alpha_10000_sub_1000' + angpostfix + '.txt').T)
    foldername = 'separate_1000'
    predictprefix = '_a'
elif dataset == 'a_10000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_alpha_10000' + angpostfix + '.txt').T)
    foldername = 'separate_10000'
    predictprefix = '_a'
elif dataset == 'b1_1000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta1_10000_sub_1000' + angpostfix + '.txt').T)
    foldername = 'separate_1000'
    predictprefix = '_b1'
elif dataset == 'b1_10000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta1_10000' + angpostfix + '.txt').T)
    foldername = 'separate_10000'
    predictprefix = '_b1'
elif dataset == 'b2_1000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta2_10000_sub_1000' + angpostfix + '.txt').T)
    foldername = 'separate_1000'
    predictprefix = '_b2'
elif dataset == 'b2_10000':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta2_10000' + angpostfix + '.txt').T)
    foldername = 'separate_10000'
    predictprefix = '_b2'
elif dataset == 'a_500':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_alpha_10000_sub_500' + angpostfix + '.txt').T)
    foldername = 'separate_500'
    predictprefix = '_a'
elif dataset == 'b1_500':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta1_10000_sub_500' + angpostfix + '.txt').T)
    foldername = 'separate_500'
    predictprefix = '_b1'
elif dataset == 'b2_500':
    # 526 x 66
    data_tensor = torch.Tensor(np.loadtxt(
        data_dir + '/dataset_beta2_10000_sub_500' + angpostfix + '.txt').T)
    foldername = 'separate_500'
    predictprefix = '_b2'
print('dataset size: {}'.format(data_tensor.size()))

kwargs = {'num_workers': 2,
          'pin_memory': True} if torch.cuda.is_available() else {}

train_loader = DataLoader(TensorDatasetDataOnly(data_tensor),
                              batch_size=batch_size,
                              shuffle=True, **kwargs)

#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)


###
##
###
#


model = VAE(args)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, x_dim=784):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False)
    BCE = F.mse_loss(recon_x, x.view(-1, x_dim), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    #for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, x_dim=model.x_dim)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def trainold(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, x_dim=model.x_dim)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test(epoch)
    sample = Variable(torch.randn(64, args.z_dim))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    #save_image(sample.data.view(64, 1, 28, 28),
    #           'results/sample_' + str(epoch) + '.png')
sample = Variable(torch.randn(2000, args.z_dim))
sample = model.decode(sample).cpu()
out = sample.data.cpu().numpy()
np.savetxt('samples_0.txt', out.T)