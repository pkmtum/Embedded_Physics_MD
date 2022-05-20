from __future__ import print_function
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributions as dist
from torchvision import datasets, transforms
import torch.nn.functional as F

from utils_peptide import convertangulardataset as convang
from utils_peptide import convertangularaugmenteddataset as convangaugmented


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
# # network for peptide dataset
# class generator(nn.Module):
#     def __init__(self, dim_data=66, dim_latent=2, dim_hidden=(10, 20, 40, 50, 50)):
#         super(generator, self).__init__()
#         self.fc1 = nn.Linear(dim_latent, dim_hidden[0])
#         self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
#         self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[2])
#         self.fc4 = nn.Linear(dim_hidden[2], dim_hidden[3])
#         self.fc5 = nn.Linear(dim_hidden[3], dim_hidden[4])
#         self.fc6 = nn.Linear(dim_hidden[4], dim_data)
#         utils.initialize_weights(self)
#
#     def forward(self, z):
#         #x = F.leaky_relu(self.fc1(z), negative_slope=0.1)
#         x = F.relu(self.fc1(z))
#         x = F.tanh(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.tanh(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         return self.fc6(x)
#
#     def forward_OLD(self, z):
#         x = F.relu(self.fc1(z))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
#
#
# class discriminator(nn.Module):
#     def __init__(self, dim_data=66, dim_hidden=(50, 50, 40, 30, 20, 10)):
#         super(discriminator, self).__init__()
#         self.fc1 = nn.Linear(dim_data, dim_hidden[0])
#         self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
#         self.fc3 = nn.Linear(dim_hidden[1], dim_hidden[2])
#         self.fc4 = nn.Linear(dim_hidden[2], dim_hidden[3])
#         self.fc5 = nn.Linear(dim_hidden[3], dim_hidden[4])
#         self.fc6 = nn.Linear(dim_hidden[4], dim_hidden[5])
#         self.fc7 = nn.Linear(dim_hidden[5], 1)
#         utils.initialize_weights(self)
#
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
#         x = F.tanh(self.fc2(x))
#         #x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
#         x = F.relu(self.fc3(x))
#         x = F.tanh(self.fc4(x))
#         x = F.leaky_relu(self.fc5(x), negative_slope=0.2)
#         x = F.tanh(self.fc6(x))
#         return F.sigmoid(self.fc7(x))
#
#     def forward_OLD(self, x):
#         x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
#         return F.sigmoid(self.fc2(x))

def printpar(modu):
    for name, param in modu.named_parameters():
        print(name)
        print(param)

class GANPeptide(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        # self.batch_size = 64
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = bool(args.gpu_mode)

        self.model_name = args.gan_type
        self.c = args.clipping              # clipping value
        self.n_critic = args.n_critic       # the number of iterations of the critic per generator iteration
        self.z_dim = args.z_dim
        self.n_samples = args.samples_pred
        self.bClusterND = bool(args.clusterND)
        self.output_postfix = args.outPostFix
        self.angulardata = args.useangulardat

        # networks init
        if self.angulardata == 'ang':
            dimdata = (22-1) * 3
            from NET_ang import generator
            from NET_ang import discriminator
        elif self.angulardata == 'ang_augmented':
            dimdata = (22-1) * 5
            from NET_angaug import generator
            from NET_angaug import discriminator
        elif self.angulardata == 'ang_auggrouped':
            dimdata = (22-1) * 5
            #from NET_angaug_long import generator
            from NET_angaug import generator
            from NET_angaug import discriminator
        else:
            dimdata = 22 * 3
            #from NET_coord_low_dim_long import generator
            from NET_coord import generator
            from NET_coord import discriminator
        # import the generator / discriminator network


        self.G = generator(dim_data=dimdata, dim_latent=self.z_dim)
        #print self.G.fc2.weight.data
        self.D = discriminator(dim_data=dimdata, boutsig=False) # False since we use BCEloss with logits
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG,
                                      betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD,
                                      betas=(args.beta1, args.beta2))

        printparams = False
        # print the parameters
        if printparams:
            print('Print initial parameters:')
            printpar(modu=self.G)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # pre-sepcify foldername variable for with dataset
        foldername = self.dataset
        predictprefix = ''

        # is using angular data set, add postfix of the data
        if self.angulardata == 'ang':
            angpostfix = '_ang'
        elif self.angulardata == 'ang_augmented':
            angpostfix = '_ang_augmented'
        elif self.angulardata == 'ang_auggrouped':
            angpostfix = '_ang_auggrouped'
        else:
            angpostfix = ''

        # load dataset
        if self.bClusterND:
            data_dir = '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide'
        else:
            data_dir = '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide'
        #data_dir = 'data/peptide'
        if self.dataset == 'm_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'samples':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_samples' + angpostfix + '.txt').T)
        elif self.dataset == 'm_526':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_526' + angpostfix + '.txt').T)
        elif self.dataset == 'm_10437':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_10537' + angpostfix + '.txt').T)
        elif self.dataset == 'a_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_alpha_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_a'
        elif self.dataset == 'a_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_alpha_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_a'
        elif self.dataset == 'b1_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta1_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_b1'
        elif self.dataset == 'b1_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta1_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_b1'
        elif self.dataset == 'b2_1000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta2_10000_sub_1000' + angpostfix + '.txt').T)
            foldername = 'separate_1000'
            predictprefix = '_b2'
        elif self.dataset == 'b2_10000':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta2_10000' + angpostfix + '.txt').T)
            foldername = 'separate_10000'
            predictprefix = '_b2'
        elif self.dataset == 'a_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_alpha_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_a'
        elif self.dataset == 'b1_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta1_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_b1'
        elif self.dataset == 'b2_500':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_beta2_10000_sub_500' + angpostfix + '.txt').T)
            foldername = 'separate_500'
            predictprefix = '_b2'
        print('dataset size: {}'.format(data_tensor.size()))

        kwargs = {'num_workers': 2,
                  'pin_memory': True} if torch.cuda.is_available() else {}
        self.data_loader = DataLoader(TensorDatasetDataOnly(data_tensor),
                                      batch_size=self.batch_size,
                                      shuffle=True, **kwargs)
        # specify as model_name the general kind of dataset: mixed or separate
        self.model_name = foldername
        self.predprefix = predictprefix

        # fixed noise
        if self.gpu_mode:
            self.sample_z_ = Variable(
                torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
        else:
            self.sample_z_ = Variable(
                torch.rand((self.batch_size, self.z_dim)), volatile=True)

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        loss_fn = nn.BCEWithLogitsLoss() #nn.BCELoss()

        kldiv = nn.KLDivLoss()

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), \
                                         Variable(torch.zeros(self.batch_size, 1).cuda())
        else:
            self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), \
                                         Variable(torch.zeros(self.batch_size, 1))

        #for name, param in self.G.named_parameters():
        #    print name
        #    print param
        self.D.train()
        # for name, param in self.D.named_parameters():
        #     print name
        #     print param

        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, x_ in enumerate(self.data_loader):
                #print(iter)
                #print(x_.size())
                #print(x_)
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    # print('here')
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = Variable(x_.cuda()), Variable(z_.cuda())
                else:
                    x_, z_ = Variable(x_), Variable(z_)

                #xnp = x_.data.numpy()

                #np.savetxt('tt.txt',xnp)

                # update D network
                self.D_optimizer.zero_grad()

                #propD = self.D(x_)
                #D_real_bernoulli = dist.Bernoulli(propD)
                #D_real_b_sample = D_real_bernoulli.sample()

                D_real = self.D(x_) #D_real_b_sample #self.D(x_)
                #print('D_real')
                #print (D_real)
                # D_real_loss = torch.mean(F.softplus(D_real))
                D_real_loss = loss_fn(D_real, self.y_real_)

                # print 'D_real_loss'
                # print D_real_loss
                D_real_loss.backward()

                G_ = self.G(z_)

                #propG = self.D(G_)
                #D_fake_bernoulli_G = dist.Bernoulli(propG)
                #D_fake_b_sample_G = D_fake_bernoulli_G.sample()

                D_fake = self.D(G_) #D_fake_b_sample_G #self.D(G_)
                #print('D_fake')
                #print (D_fake)
                # D_fake_loss = torch.mean(F.softplus(-D_fake))
                # D_fake_loss = loss_fn(D_fake, Variable(torch.zeros(self.batch_size, 1)))
                D_fake_loss = loss_fn(D_fake, self.y_fake_)

                D_fake_loss.backward()

                # print 'D_fake_loss'
                # print D_fake_loss
                # print D_fake_loss
                D_loss = D_real_loss + D_fake_loss
                # D_loss.backward()

                #D_loss.backward()
                self.D_optimizer.step()

                # clipping D
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.c, self.c)

                # if ((iter+1) % self.n_critic) == 0:
                    # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)

                #propGfake2 = self.D(G_)
                #D_fake_bernoulli_Gfake = dist.Bernoulli(propGfake2)
                #D_fake_b_sample_Gfake = D_fake_bernoulli_Gfake.sample()

                D_fake2 = self.D(G_) #D_fake_b_sample_Gfake#self.D(G_)
                G_loss = loss_fn(D_fake2, self.y_real_)
                #G_loss = loss_fn(D_fake2, Variable(torch.ones(self.batch_size, 1)))

                self.train_hist['G_loss'].append(G_loss.data[0])

                G_loss.backward()
                self.G_optimizer.step()

                self.train_hist['D_loss'].append(D_loss.data[0])

                if ((iter + 1) % 25) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.data[0], G_loss.data[0]))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            # self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # print the parameters
        # print('Print parameters after training:')
        #for name, param in self.G.named_parameters():
        #    print name
        #    print param
        # for name, param in self.D.named_parameters():
        #     print name
        #     print param
        self.save()
        self.gen_samples(self.n_samples)
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                          self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name, self.output_postfix), self.model_name + self.predprefix)

    def gen_samples(self, n_samples):

        # print parameters for debugging purposes



        #print self.G.fc2.weight.data
        self.G.eval()
        # this was the for the old case
        #save_dir = self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.output_postfix
        save_dir = self.result_dir + '/' + self.model_name + '/' + self.output_postfix

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.gpu_mode:
            sample_z_ = Variable(
                torch.rand((n_samples, self.z_dim)).cuda(), volatile=True)
        else:
            sample_z_ = Variable(torch.rand((n_samples, self.z_dim)),
                                 volatile=True)

        samples = self.G(sample_z_).data.cpu().numpy()

        # visualize mapping between the different layers
        if hasattr(self.G, 'plotlatentsamples'):
            self.G.plotlatentsamples(n_samples=100, z_dim=self.z_dim)
        else:
            print('No visualization in generator available.')

        # convert the samples if they are in the angular format
        if self.angulardata == 'ang':
            samplesout = convang(samples.T)
        elif self.angulardata == 'ang_augmented':
            samplesout = convangaugmented(samples.T)
        elif self.angulardata == 'ang_auggrouped':
            samplesout = convangaugmented(samples.T, bgrouped=True )
        else:
            samplesout = samples.T

        np.savetxt(save_dir + '/samples' + self.predprefix + '.txt', samplesout)
        print(np.amax(samples))
        print(np.amin(samples))
        print(np.mean(samples))
        print(np.std(samples))
        print('Done generating {} samples'.format(self.n_samples))

        real_data = np.loadtxt('../../data_peptide/dataset_alpha_10000_sub_1000.txt').T
        print(np.amax(real_data))
        print(np.amin(real_data))
        print(np.mean(real_data))
        print(np.std(real_data))




    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.output_postfix):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.output_postfix)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            if self.gpu_mode:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)).cuda(), volatile=True)
            else:
                sample_z_ = Variable(torch.rand((self.batch_size, self.z_dim)), volatile=True)

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.output_postfix + '/'
                          + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, self.output_postfix)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + self.predprefix + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + self.predprefix + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + self.predprefix + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name, self.output_postfix)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + self.predprefix + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + self.predprefix + '_D.pkl')))
