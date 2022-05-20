import utils, torch, time, os, pickle, datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributions as dist
from torchvision import datasets, transforms
import torch.nn.functional as F

import math

from utils_peptide import convertangulardataset as convang
from utils_peptide import convertangularaugmenteddataset as convangaugmented
from utils_peptide import convert_given_representation
#from utils_peptide_torch import register_nan_checks

# Import classes related to MD simulations
from MDLoss import MDLoss
from MDLoss import MDSimulator

from GaussianRefModelParametrization import GaussianRefModelParametrization as GaussRefParams

class MVN:
    def __init__(self, mean, cov):
        self.mean = mean.copy()
        self.cov = cov.copy()
    def sample(self):
        return np.random.multivariate_normal(self.mean, self.cov)

class UQ:
    def __init__(self, bdouq=False, bcalchess=False, blayercov=False, buqbias=False):
        self.bdouq = bdouq
        self.npostsamples = 100
        self.bhessavailable = bcalchess
        self.blayercov = blayercov
        self.buqbias = buqbias

def checkandcreatefolder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        diraug = dir
    else:
        datetimepostfix = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        diraug = os.path.join(dir, datetimepostfix)
        os.makedirs(diraug)
    return diraug


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


def model_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        #print(module)
        bnans = False
        if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            bnans = True
            print module
            print('NaN gradient in ' + type(module).__name__)
        return bnans

class VARjoint(object):
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
        self.gpu_mode = bool(args.gpu_mode) and torch.cuda.is_available()

        self.model_name = args.gan_type
        self.c = args.clipping  # clipping value
        self.n_critic = args.n_critic  # the number of iterations of the critic per generator iteration
        self.z_dim = args.z_dim
        self.n_samples = args.samples_pred
        self.bClusterND = bool(args.clusterND)
        self.output_postfix = args.outPostFix
        self.angulardata = args.useangulardat
        self.autoencvarbayes = bool(args.AEVB)
        self.L = args.L # amount of eps ~ p(eps) = N(0,1)
        self.Z = args.Z # amount of samples from p(z)
        self.outputfrequ = args.outputfreq
        self.n_samples_per_mu = args.samples_per_mean  # if 0, just use mean prediction: x = mu(z)
        self.lambdaexpprior = args.exppriorvar

        self.exactlikeli = bool(args.exactlikeli)

        bqu = bool(args.npostS)
        self.uqoptions = UQ(bdouq=bqu, bcalchess=True, blayercov=False, buqbias=bool(args.uqbias))
        self.uqoptions.npostsamples = args.npostS
        self.bfixlogvar = bool(args.sharedlogvar)

        # check if a trained model should be loaded
        self.filemodel = args.loadtrainedmodel
        self.bloadmodel = bool(self.filemodel)

        self.bvislatent_training = True
        self.bvismean_and_samples = False

        self.bassigrandW = bool(args.assignrandW)
        self.bfreememory = bool(args.freeMemory)

        # select the forward model
        self.x_dim = args.x_dim
        self.joint_dim = self.x_dim + self.z_dim

        self.coordinatesiunit = 1.e-9
        self.coorddataprovided = 1.e-10
        self.bDebug = False
        self.bCombinedWithData = False


        # import the reference model
        nModes = 2

        # ARD prior
        if args.ard > 0.:
            self.bard = True
            self.arda0 = args.ard
        else:
            self.bard = False
            self.arda0 = 0.

        # we can only sample if p(x|z) is a Gaussian: N(mu(z), sigmasq(z))
        if not self.autoencvarbayes:
            self.n_samples_per_mu = 0

        # seed the calculation if required
        if not args.seed == 0:
            torch.manual_seed(args.seed)
            if bool(args.gpu_mode):
                torch.cuda.manual_seed(args.seed)

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

        # specify peptide name
        self.name_model = 'ala_2'
        self.name_peptide = 'ala_2'

        # load dataset
        if self.bClusterND:
            data_dir = '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide'
        else:
            data_dir = '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide'
        self.data_dir = data_dir
        # data_dir = 'data/peptide'
        if self.dataset == 'm_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_1527' + angpostfix + '.txt').T)
            self.N = 1527
        elif self.dataset == 'b1b2_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_b1b2_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'ab1_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_ab1_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'ab2_1527':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_ab2_1527' + angpostfix + '.txt').T)
        elif self.dataset == 'm_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_4004' + angpostfix + '.txt').T)
            self.N = 4004
        elif self.dataset == 'm_102':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_102' + angpostfix + '.txt').T)
            self.N = 102
        elif self.dataset == 'm_262':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_262' + angpostfix + '.txt').T)
            self.N = 262
        elif self.dataset == 'm_52':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_52' + angpostfix + '.txt').T)
            self.N = 52
        elif self.dataset == 'ma_10':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_10' + angpostfix + '.txt').T)
            self.N = 10
        elif self.dataset == 'ma_50':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_50' + angpostfix + '.txt').T)
            self.N = 50
        elif self.dataset == 'ma_100':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_100' + angpostfix + '.txt').T)
            self.N = 100
        elif self.dataset == 'ma_200':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_200' + angpostfix + '.txt').T)
            self.N = 200
        elif self.dataset == 'ma_500':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_500' + angpostfix + '.txt').T)
            self.N = 500
        elif self.dataset == 'ma_1000':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_1000' + angpostfix + '.txt').T)
            self.N = 1000
        elif self.dataset == 'ma_1500':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_1500' + angpostfix + '.txt').T) # / (self.coordinatesiunit / self.coorddataprovided))
            self.N = 1500
        elif self.dataset == 'ma_4000':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_4000' + angpostfix + '.txt').T)
            self.N = 4000
        elif self.dataset == 'ma_13334':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_13334' + angpostfix + '.txt').T)
            self.N = 13334
        elif self.dataset == 'b1b2_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_b1b2_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'ab1_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_ab1_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'ab2_4004':
            # 1527 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_ab2_4004' + angpostfix + '.txt').T)
        elif self.dataset == 'samples':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_samples' + angpostfix + '.txt').T)
        elif self.dataset == 'm_526':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_526' + angpostfix + '.txt').T)
            self.N = 526
        elif self.dataset == 'm_1001':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_mixed_1001' + angpostfix + '.txt').T)
            self.N = 1001
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
        elif self.dataset == 'm_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/dataset_ala_15' + angpostfix + '.txt').T)
            self.N = 2000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_100_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_100' + angpostfix + '.txt').T)
            self.N = 100
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_200_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_200' + angpostfix + '.txt').T)
            self.N = 200
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_300_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_300' + angpostfix + '.txt').T)
            self.N = 300
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_500_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_500' + angpostfix + '.txt').T)
            self.N = 500
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_1500_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_1500' + angpostfix + '.txt').T)
            self.N = 1500
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_3000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_3000' + angpostfix + '.txt').T)
            self.N = 3000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_5000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_5000' + angpostfix + '.txt').T)
            self.N = 5000
            self.name_peptide = 'ala_15'
        elif self.dataset == 'm_10000_ala_15':
            # 526 x 66
            data_tensor = torch.Tensor(np.loadtxt(
                data_dir + '/ala-15/dataset_ala_15_10000' + angpostfix + '.txt').T)
            self.N = 10000
            self.name_peptide = 'ala_15'

        # categorize what to do. Combine data and reverse variational approach or not
        if 'ala_15' in self.name_peptide:
            print 'We do not support ALA15 peptide in the current version.'
            quit()
        elif (self.dataset == 'var_gauss' or self.dataset == 'ala_2'):
            self.bCombinedWithData = False
            self.N = 0
            # specify the model name and prefix
            if self.dataset == 'var_gauss':
                self.name_model = 'gauss'
                predictprefix = '_gauss'
            elif self.dataset == 'ala_2':
                self.name_model = 'ala_2'
                predictprefix = '_ala_2'

        # in this case we combine the ala_2 reverse variational model with VAE
        else:
            self.bCombinedWithData = True
            self.name_model = 'ala_2'

        if not (self.dataset == 'var_gauss' or self.dataset == 'ala_2'):
            print('dataset size: {}'.format(data_tensor.size()))

            self.kwargsdatloader = {'num_workers': 2,
                                    'pin_memory': True} if torch.cuda.is_available() else {}

            self.data_tensor = data_tensor
            self.data_loader = DataLoader(TensorDatasetDataOnly(data_tensor),
                                          batch_size=self.batch_size,
                                          shuffle=True, **self.kwargsdatloader)

            # for visualization purposes
            if self.dataset == 'm_1527':
                self.data_tensor_vis_1527 = self.data_tensor
            elif 'ala_15' not in self.dataset:
                self.data_tensor_vis_1527 = torch.Tensor(
                    np.loadtxt(data_dir + '/dataset_mixed_1527' + angpostfix + '.txt').T )#/ (self.coordinatesiunit / self.coorddataprovided))
            elif 'ala_15' in self.dataset:
                self.data_tensor_vis_1527 = torch.Tensor(
                    np.loadtxt(data_dir + '/ala-15/dataset_ala_15_1500' + angpostfix + '.txt').T )#/ (self.coordinatesiunit / self.coorddataprovided))

        # specify as model_name the general kind of dataset: mixed or separate
        self.predprefix = predictprefix

        # saving directory
        tempdir = os.path.join(self.result_dir, self.model_name, foldername, self.output_postfix)
        self.output_dir = checkandcreatefolder(dir=tempdir)

        if self.name_model is not 'var_gauss':
            from MDLoss import MDSimulator as ReferenceModel
            self.MDLossapplied = MDLoss.apply

            if 'ang' in self.angulardata:
                from VARJmodel import VARmdAngAugGrouped as VARmod
            else:
                #from VARJmodel import VARmd as VARmod
                from VAEmodelKingma import VAEmod as VARmod
        else:
            if nModes == 1:
                from VARJmodel import ReferenceModel as ReferenceModel
                from VARJmodel import VARmod as VARmod
            else:
                from VARJmodel import ReferenceModelMultiModal as ReferenceModel
                from VARJmodel import VARmixture as VARmod
                #from VARJmodel import VARmixturecomplex as VARmod

        # initialize the reference model
        if self.name_model is not 'var_gauss':
            if self.bClusterND:
                reffolderPDB = '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide/filesALA2/reftraj/'
            else:
                reffolderPDB = '/home/schoeberl/Dropbox/PhD/projects/2018_07_06_openmm/ala2/'

            self.refmodel = MDSimulator(os.path.join(reffolderPDB, 'ala2_adopted.pdb'), bGPU=self.gpu_mode, sAngularRep=self.angulardata, sOutputpath=self.output_dir)
        else:
            # specify reference model (onyl needed if not MD run)
            muref, sigmaref, W_ref = GaussRefParams.getParVectors(x_dim=self.x_dim, z_dim=self.z_dim, nModes=nModes,
                                                                  bassigrandW=self.bassigrandW)
            self.refmodel = ReferenceModel(mu=muref, sigma=sigmaref, W=W_ref, outputdir=self.output_dir,
                                           bgpu=self.gpu_mode)
            self.refmodel.plot(path=self.output_dir)

        # initialize the model
        ###################################################################
        self.vaemodel = VARmod(args, self.x_dim, self.bfixlogvar)
        ###################################################################

        # check the gradients for nans
        #register_nan_checks(self.vaemodel)

        if self.gpu_mode:
            self.vaemodel.cuda()

        # initialize the optimizer
        self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)

    def getweightlist(self):
        weight_list = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                weight_list.append({'name': name, 'id': id, 'params': param})
                #pclone = param.clone()
                #params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                print(name)  # , param.data
            id = id + 1
        return weight_list

    def storeweightlist(self, parlist, path, prefix=None, postfix=None):

        folder = os.path.join(path, prefix)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        for paritem in parlist:
            temp = paritem['params'].data
            np.savetxt(os.path.join(folder, paritem['name']), temp.cpu().numpy())

    def getdecweightlist(self):
        decoding_weight_list = []
        params_dec_copy = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if 'dec_' in name:
                    # check if we want to uq bias uncertainty
                    if not ('.bias' in name) and not ('logvar' in name):
                        decoding_weight_list.append({'name': name, 'id': id, 'params': param})
                        #pclone = param.clone()
                        #params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print name  # , param.data
                    id = id + 1
        return decoding_weight_list


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_autoencvarbayes(self, recon_mu, recon_logvar, x, mu, logvar, x_dim=784, normalize=False):

        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False)
        pointwiseMSEloss = 0.5 * F.mse_loss(recon_mu, x.view(-1, x_dim), size_average=False, reduce=False)

        # Maug is here the augmentet bacht size: explicitly: M*L while L is the amount of sample for \epsilon ~ p(\epsilon)
        Maug = pointwiseMSEloss.shape[0]

        sigsq = recon_logvar.exp()
        # np.savetxt('var.txt', sigsq.data.cpu().numpy())
        weight = sigsq.reciprocal()  # 1./sigsq

        logvarobjective = 0.5 * recon_logvar.sum()

        pointwiseWeightedMSEloss = pointwiseMSEloss.mul(weight)
        WeightedMSEloss = pointwiseWeightedMSEloss.sum()

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        #l = logvar.data.cpu().numpy()
        #np.savetxt('var.txt', l)

        self.train_hist['kl_qp'].append(KLD)

        # Prior on predictive variance
        psigsqlamb = self.lambdaexpprior
        # employ prior if desired
        if psigsqlamb > 0.:
            lamb = torch.FloatTensor(1)
            lamb.fill_(psigsqlamb)

            if self.gpu_mode:
                lambvariable = Variable(lamb.cuda())
            else:
                lambvariable = Variable(lamb)
            loglamb = lambvariable.log()

            # minus here becuase of minimization; expression stems from max log-likelihood
            logpriorpvarexpanded = - (loglamb.expand_as(sigsq) - sigsq.mul(psigsqlamb))

            logpriorvarsum = logpriorpvarexpanded.sum()
            logpriorvar = logpriorvarsum.div(Maug)
        else:
            logpriorvar = torch.zeros_like(KLD)

        # return (WeightedMSEloss + KLD)
        loss = (logvarobjective + WeightedMSEloss + KLD + logpriorvar)

        if self.bard:
            ardcontrib = self.ardprior.getlogpiorARD()
            ardcontrib.mul_(float(Maug)/self.N)
            loss.add_(-ardcontrib[0])

        # normalize for summing to second part of loss
        if normalize:
            loss.div_(float(Maug))

        ## TODO add here actually the single contribution explicitly
        #nancheck = torch.tensor([loss])
        #nans = torch.isnan(nancheck)
        #nanentries = nans.nonzero()
        #if nanentries.nelement() > 0:
        #    print nancheck

        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_variationalmodel(self, qmu, qlogvar, x_data, z_data, pmu, plogvar, N_z, N_zpx, x_dim=784, bgpu=False, normalize=False):
        bcov = True

        # < < log p(x) >_p(x|z) >_p(z)

        #if bgpu:
        #    covinvref = covinvref.cuda()
        #    covref = covref.cuda()

        #if self.refmodel.getIsMixture():
        if self.refmodel.getModelType() is 'GaussianMixture':

            mixtureweight = 0.5
            nmixture = self.refmodel.getMixtures()
            muref = self.refmodel.getmu()

            if bgpu:
                x_i = torch.zeros(x_data.shape[0], nmixture).cuda()
            else:
                x_i = torch.zeros(x_data.shape[0], nmixture)

            for i in range(0, nmixture):

                covinvref = self.refmodel.getInvCov(i)
                covref = self.refmodel.getCov(i)

                if bgpu:
                    murefexpanded = muref[i, :].expand(x_data.shape[0], x_data.shape[1]).cuda()
                else:
                    murefexpanded = muref[i, :].expand(x_data.shape[0], x_data.shape[1])

                xmmu = x_data - murefexpanded
                if self.bfreememory:
                    del murefexpanded

                xmmucovinv = torch.mm(xmmu, covinvref)
                xcovinvx = torch.sum(torch.mul(xmmucovinv, xmmu), dim=1)

                log2pi = np.log(2 * math.pi)

                x_i[:, i] = xcovinvx.mul(-0.5).add(-0.5*self.x_dim*log2pi + np.log(mixtureweight))

                # TODO Check if there is covref.logdet() available. This could cause numerical instabilities.
                x_i[:, i] -= 0.5 * covref.det().log()

                if self.bfreememory:
                    del xcovinvx

                #if bgpu:
                #    x_i[i] -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
                #else:
                #    x_i[i] -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx

                #x_i[i] -= 0.5 * covref.det().log()

            m, m_pos = x_i.max(dim=1, keepdim=True)
            xma = x_i - m
            expxma = torch.exp(xma)
            sumexpxma = expxma.sum(dim=1, keepdim=True)
            logsumtemp = torch.log(sumexpxma)
            logsumtemppm = m + logsumtemp
            logpx = logsumtemppm.sum()

        elif self.refmodel.getModelType() is 'MD':
            #print 'Not implemented so far. Try another time.'
            #quit()
            logpx = self.MDLossapplied(x_data, self.refmodel, True)


            #logpx = x_i.exp().sum().log()
        elif self.refmodel.getModelType() is 'Gaussian':
            muref = self.refmodel.getmu()
            covinvref = self.refmodel.getInvCov()
            covref = self.refmodel.getCov()

            if bgpu:
                murefexpanded = muref.expand(x_data.shape[0], x_data.shape[1]).cuda()
            else:
                murefexpanded = muref.expand(x_data.shape[0], x_data.shape[1])

            if bcov:
                xmmu = x_data - murefexpanded
                if self.bfreememory:
                    del murefexpanded

                xmmucovinv = torch.mm(xmmu, covinvref)
                xcovinvx = torch.mul(xmmucovinv, xmmu)
                logpx = - 0.5 * xcovinvx.sum()
                logvarpx = covref.det().log()
            else:
                pointwiseLogpx = -0.5 * F.mse_loss(x_data, murefexpanded, size_average=False, reduce=False)
                if self.bfreememory:
                    del murefexpanded
                pxSigma = self.refmodel.getSigma()
                sgima = pxSigma
                if bgpu:
                    sgima = sgima.cuda()
                sigsq = torch.mul(sgima, sgima)
                ## np.savetxt('var.txt', sigsq.data.cpu().numpy())
                if bgpu:
                    weight = sigsq.reciprocal().cuda()  # 1./sigsq
                else:
                    weight = sigsq.reciprocal()  # 1./sigsq
                weightexpanded = weight.expand(x_data.shape[0], x_data.shape[1])

                #logvarobjective = 0.5 * recon_logvar.sum()

                pointwiseWeightedMSEloss = pointwiseLogpx.mul(weightexpanded)
                if self.bfreememory:
                    del pointwiseWeightedMSEloss ,weightexpanded, weight

                logpx = pointwiseWeightedMSEloss.sum() #xcovinvx.sum()
                logvarpx = sigsq.log().sum()

            #logpx -= 0.5 * logvarpx * N_z * N_zpx
            if bgpu:
                logpx -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
            else:
                logpx -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx
        else:
            print 'Not implemented so far. Try another time.'
            quit()

        # < < log q(z|x) >_p(x|z) >_p(z)
        pointwiseLogqzgx = -0.5 * F.mse_loss(z_data, qmu, size_average=False, reduce=False)
        weightqzgx = qlogvar.exp().reciprocal() # sigsq.reciprocal()  # 1./sigsq
        weightqzgxexpanded = weightqzgx.expand(z_data.shape[0], z_data.shape[1])

        #logvarobjective = 0.5 * recon_logvar.sum()

        pointwiseWeightedMSElossLogqzgx = pointwiseLogqzgx.mul(weightqzgxexpanded)
        logqzgx = pointwiseWeightedMSElossLogqzgx.sum()
        if self.bfreememory:
            del pointwiseWeightedMSElossLogqzgx, weightqzgxexpanded, pointwiseLogqzgx

        logqzgx -= 0.5 * qlogvar.sum()
        if bgpu:
            logqzgx -= self.z_dim * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
        else:
            logqzgx -= self.z_dim * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx
        #logqzgx -= 0.5 * (torch.ones_like(z_data)).mul(2 * math.pi)

        # < < log p(x|z) >_p(x|z) >_p(z)
        pointwiseLogpxgz = -0.5 * F.mse_loss(x_data, pmu, size_average=False, reduce=False)
        weightpxgz = plogvar.exp().reciprocal() # sigsq.reciprocal()  # 1./sigsq
        weightpxgzexpanded = weightpxgz.expand(x_data.shape[0], x_data.shape[1])

        pointwiseWeightedMSElossLogpxgz = pointwiseLogpxgz.mul(weightpxgzexpanded)
        logpxgz = pointwiseWeightedMSElossLogpxgz.sum()
        if self.bfreememory:
            del pointwiseWeightedMSElossLogpxgz, pointwiseLogpxgz, weightpxgzexpanded

        logpxgz -= 0.5 * plogvar.sum()
        if bgpu:
            logpxgz -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
        else:
            logpxgz -= self.x_dim * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx


        # < < log p(z) >_p(x|z) >_p(z)
        pointwiseLogpz = -0.5 * F.mse_loss(z_data, torch.zeros_like(z_data), size_average=False, reduce=False)
        weightpz = torch.ones_like(z_data)
        weightpzexpanded = weightpz.expand(z_data.shape[0], z_data.shape[1])

        pointwiseWeightedMSElossLogpz = pointwiseLogpz.mul(weightpzexpanded)
        logpz = pointwiseWeightedMSElossLogpz.sum()
        #logpz -= 0.5 * torch.ones_like(z_data).log().sum()
        if bgpu:
            logpz -= self.z_dim * 0.5 * torch.tensor(2 * math.pi).cuda().log() * N_z * N_zpx
        else:
            logpz -= self.z_dim * 0.5 * torch.tensor(2 * math.pi).log() * N_z * N_zpx

        nancheck = torch.tensor([logqzgx, logpx, logpxgz, logpz])

        nans = torch.isnan(nancheck)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            print nancheck

        loss = - logqzgx - logpx + logpxgz + logpz

        if normalize:
            loss.div_(float(N_z * N_zpx))

        return loss

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 #* torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.train_hist['kl_qp'].append(KLD)

        # Prior on predictive variance

        psigsqlamb = self.lambdaexpprior
        # employ prior if desired
        if psigsqlamb > 0.:
            lamb = torch.FloatTensor(1)
            lamb.fill_(psigsqlamb)

            if self.gpu_mode:
                lambvariable = Variable(lamb.cuda())
            else:
                lambvariable = Variable(lamb)
            loglamb = lambvariable.log()

            # minus here becuase of minimization; expression stems from max log-likelihood
            logpriorpvarexpanded = - (loglamb.expand_as(sigsq) - sigsq.mul(psigsqlamb))
            N = logpriorpvarexpanded.size()[0]
            logpriorvarsum = logpriorpvarexpanded.sum()
            logpriorvar = logpriorvarsum.div(N)
        else:
            logpriorvar = Variable(torch.zeros_like(KLD))

        # return (WeightedMSEloss + KLD)
        #loss = (logvarobjective + WeightedMSEloss + KLD + logpriorvar)
        loss = (logpx)
        lossnp = loss.data.cpu().numpy()
        if lossnp != lossnp:
            print('Error: Loss is NaN')
        return loss

    def nanCheck(self, input, name):
        nans = torch.isnan(input)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            print name
            print input
            return True
        else:
            return False

    def trainepochCombined(self, epoch, weight_vae):

        N_z = self.Z
        N_xpz = self.L

        self.vaemodel.train()

        # for batch_idx, (data, _) in enumerate(train_loader):
        # batch_idx could be replaced by iter (iteration during batch)
        for batch_idx, data_x_vae in enumerate(self.data_loader):

            # if (1.-weight_vae) > 0.:
            #     # create 'data' samples from p_theta(z)
            #     # with torch.no_grad():
            #     data_z = torch.randn((N_z, self.z_dim))
            #
            #     # copy the data tensor for using more eps ~ p(eps) samples Eq. (7) in AEVB paper
            #     dataaug = data_z.repeat(self.L, 1)
            #     data_z = Variable(dataaug)
            #     data_z.requires_grad = True
            #
            #     if self.gpu_mode:
            #         data_z = data_z.cuda()
            #
            #     self.optimizer.zero_grad()
            #
            #     # ,mu and logvar of p(x|z)
            #     data_x_rev_variational, q_recon_batch, pmu, plogvar = self.vaemodel.forward(data_z)
            #     qmu = q_recon_batch[0]
            #     qlogvar = q_recon_batch[1]
            #
            #     if self.bDebug:
            #         bNaN = np.zeros(6, dtype=bool)
            #         bNaN[0] = self.nanCheck(input=data_z, name='data_z')
            #         bNaN[1] = self.nanCheck(input=data_x_rev_variational, name='data_x')
            #         bNaN[2] = self.nanCheck(input=qmu, name='qmu')
            #         bNaN[3] = self.nanCheck(input=qlogvar, name='qlogvar')
            #         bNaN[4] = self.nanCheck(input=pmu, name='pmu')
            #         bNaN[5] = self.nanCheck(input=plogvar, name='plogvar')
            #         if np.any(bNaN):
            #             print 'NaN occurring.'
            #
            #     # print np.exp(recon_logvar.data.cpu().numpy())
            #     loss_rev_variational = self.loss_function_variationalmodel(qmu, qlogvar, data_x_rev_variational, data_z, pmu, plogvar, N_z, N_xpz,
            #                                                x_dim=self.x_dim, bgpu=self.gpu_mode, normalize=True)

            #####################
            # VAE PART
            #####################

            #dataaug_vae = Variable(data_x_vae.repeat(self.L, 1))
            #if self.gpu_mode:
            #    dataaug_vae = dataaug_vae.to('cuda')

            L = self.L
            dataaug = data_x_vae.clone()
            for l in xrange(L - 1):
                dataaug = torch.cat((dataaug, data_x_vae), 0)
            data_x_vae = Variable(dataaug)
            if self.gpu_mode:
                data_x_vae = data_x_vae.cuda()

            recon_batch_vae, mu_vae, logvar_vae = self.vaemodel(data_x_vae) #.forward_vae(data_x_vae)
            recon_mu_vae = recon_batch_vae[0]
            recon_logvar_vae = recon_batch_vae[1]
            #print recon_mu_vae, recon_logvar_vae, mu_vae, logvar_vae
            #quit()

            loss = self.loss_function_autoencvarbayes(recon_mu_vae, recon_logvar_vae, data_x_vae, mu_vae, logvar_vae, x_dim=self.x_dim, normalize=False)

            #if (1.-weight_vae) > 0:
            #    loss = loss_rev_variational.mul_(1.-weight_vae) + loss_vae.mul_(weight_vae)
            #else:
            #loss = loss_vae

            # loss.backward(retain_graph=True)
            loss.backward()
            self.optimizer.step()

            # print torch.autograd.grad(loss, data_x, retain_graph=True)
            # # get single grad of x val:
            # for group in self.optimizer.param_groups:
            #    for p in group['params']:
            #        g = torch.autograd.grad(data_x, p, retain_graph=True)

            # if p.grad is None:
            #    continue
            # grad = p.grad.data
            if (1. - weight_vae) > 0:
                nn.utils.clip_grad_value_(self.vaemodel.parameters(), 1.e10)

            ## check for nans only if we are in debugging modus. otherwise save the time.
            #if self.bDebug:
            #    if not model_nan_checks(self.vaemodel):
            #        self.optimizer.step()
            #    else:
            #        print 'NaN in gradient calculation.'
            #        self.vaemodel.zero_grad()
            #else:
            #    self.optimizer.step()

            log_interval = 20
            if batch_idx % log_interval == 0:
                print loss.data[0] / len(data_x_vae)
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(epoch, batch_idx, 100. * batch_idx / N_xpz * N_z, loss.data[0] / len(data_x_vae)))

        self.train_hist['Total_loss'].append(loss.data[0] / len(data_x_vae))
        #self.train_hist['Total_loss'].append(loss.data[0] / N_xpz * N_z)

    def trainepoch(self, epoch):

        N_z = self.Z
        N_xpz = self.L

        self.vaemodel.train()

        # for batch_idx, (data, _) in enumerate(train_loader):
        # batch_idx could be replaced by iter (iteration during batch)
        #for batch_idx, data in enumerate(self.data_loader):
        for batch_idx in range(0, 1):

            # create 'data' samples from p_theta(z)
            #with torch.no_grad():
            data_z = torch.randn((N_z, self.z_dim))

            # copy the data tensor for using more eps ~ p(eps) samples Eq. (7) in AEVB paper
            L = self.L
            dataaug = data_z.clone()
            for l in xrange(L - 1):
                dataaug = torch.cat((dataaug, data_z), 0)

            data_z = Variable(dataaug)
            data_z.requires_grad = True

            if self.gpu_mode:
                data_z = data_z.cuda()

            self.optimizer.zero_grad()

            if self.autoencvarbayes:
                # ,mu and logvar of p(x|z)
                data_x, q_recon_batch, pmu, plogvar = self.vaemodel(data_z)
                qmu = q_recon_batch[0]
                qlogvar = q_recon_batch[1]

                if self.bDebug:
                    bNaN = np.zeros(6, dtype=bool)
                    bNaN[0] = self.nanCheck(input=data_z, name='data_z')
                    bNaN[1] = self.nanCheck(input=data_x, name='data_x')
                    bNaN[2] = self.nanCheck(input=qmu, name='qmu')
                    bNaN[3] = self.nanCheck(input=qlogvar, name='qlogvar')
                    bNaN[4] = self.nanCheck(input=pmu, name='pmu')
                    bNaN[5] = self.nanCheck(input=plogvar, name='plogvar')
                    if np.any(bNaN):
                        print 'NaN occurring.'

                # print np.exp(recon_logvar.data.cpu().numpy())
                loss = self.loss_function_variationalmodel(qmu, qlogvar, data_x, data_z, pmu, plogvar, N_z, N_xpz, x_dim=self.x_dim, bgpu=self.gpu_mode)
            else:
                print 'This approach is based on reparametrization.'
                quit()

            #loss.backward(retain_graph=True)
            loss.backward()

            #print torch.autograd.grad(loss, data_x, retain_graph=True)
            # # get single grad of x val:
            #for group in self.optimizer.param_groups:
            #    for p in group['params']:
            #        g = torch.autograd.grad(data_x, p, retain_graph=True)

                #if p.grad is None:
                #    continue
                #grad = p.grad.data

            nn.utils.clip_grad_value_(self.vaemodel.parameters(), 1.e10)
            # check for nans only if we are in debugging modus. otherwise save the time.
            if self.bDebug:
                if not model_nan_checks(self.vaemodel):
                    self.optimizer.step()
                else:
                    print 'NaN in gradient calculation.'
                    self.vaemodel.zero_grad()
            else:
                self.optimizer.step()

            log_interval = 20
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}'.format(
                    epoch, batch_idx * N_xpz*N_z, N_xpz*N_z,
                           100. * batch_idx / N_xpz*N_z,
                           loss.data[0] / N_xpz*N_z))

        self.train_hist['Total_loss'].append(loss.data[0] / N_xpz*N_z)

    def samplePosterior(self, npostsamples):

        pert_param_list, params_dec_copy, hess_list = self.getHessian()

        #quit()

        list_normals = []
        id = 0
        for hess in hess_list:
            if not hess['fullcov']:
                var = hess['diaghessian'].reciprocal()
                scale = var.sqrt()
                mean = params_dec_copy[id]['params']
                list_normals.append(torch.distributions.Normal(mean.data, scale))
            else:
                if not hess['parisvector']:
                    shape = params_dec_copy[id]['params'].shape()
                    elements = shape[0]*shape[1]
                    mean = params_dec_copy[id]['params'].reshape(elements)
                    cov = hess['diaghessian'].inverse()
                    meannp = mean.data.cpu().numpy()
                    covnp = cov.cpu().numpy()
                    list_normals.append(MVN(meannp, covnp))
                else:
                    mean = params_dec_copy[id]['params']
                    cov = hess['diaghessian'].inverse()
                    meannp = mean.data.cpu().numpy()
                    covnp = cov.cpu().numpy()
                    list_normals.append(MVN(meannp, covnp))
            id += 1

        for i in range(npostsamples):
            id = 0
            for parlistitem in pert_param_list:
                hess = hess_list[id]
                if not (parlistitem['name'] == 'dec_logvar'):
                    if not hess['fullcov']:
                        sample = list_normals[id].sample()
                        parlistitem['params'].data.set_(sample)
                    else:
                        if not hess['parisvector']:
                            shape = params_dec_copy[id]['params'].shape()

                            samplenp = list_normals[id].sample()
                            samplevec = torch.from_numpy(samplenp)
                            sample = samplevec.resize(shape[0], shape[1])
                            parlistitem['params'].data.set_(sample)

                        else:
                            samplenp = list_normals[id].sample()
                            sample = torch.from_numpy(samplenp).float()
                            parlistitem['params'].data.set_(sample)

                id += 1

            # make predictions with posterior samples \theta ~ p(\theta | Data)
            self.gen_samples(n_samples=self.n_samples, postsampid=i)

    def getHessian(self):

        blayercov = self.uqoptions.blayercov
        self.uqoptions.bhessavailable = True

        bvartemp = self.vaemodel.bgetlogvar
        self.vaemodel.bgetlogvar = True

        data_loader_hessian_approx = DataLoader(TensorDatasetDataOnly(self.data_tensor), batch_size=self.N,
                                                batch_sampler=None,
                                                shuffle=False, **self.kwargsdatloader)

        for index, data in enumerate(data_loader_hessian_approx):
            data = Variable(data)
            if self.gpu_mode:
                data = data.cuda()

            # resetting any gradient
            self.optimizer.zero_grad()

            if self.autoencvarbayes:
                recon_batch, mu, logvar = self.vaemodel(data)
                recon_mu = recon_batch[0]
                recon_logvar = recon_batch[1]

                # print np.exp(recon_logvar.data.cpu().numpy())
                loss = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, data, mu, logvar, x_dim=self.x_dim)
            else:
                recon_batch, mu, logvar = self.vaemodel(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, x_dim=self.x_dim)

            # calculate gradient
            loss.backward(retain_graph=True)

        # identify the parameters to be pertubed / hessin calculated
        pert_param_list = []
        params_dec_copy = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if 'dec_' in name:
                    # check if we want to uq bias uncertainty
                    if '.bias' in name and self.uqoptions.buqbias:
                        pert_param_list.append({'name': name, 'id': id, 'params': param})
                        pclone = param.clone()
                        params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print name  # , param.data
                    else:
                        pert_param_list.append({'name': name, 'id': id, 'params': param})
                        pclone = param.clone()
                        params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print name  # , param.data
                    id = id + 1

        hess_list = []

        # for group in param_groups:
        #    for p in group['params'][2*nLayerEnc:]:
        for parentry in pert_param_list:
            p = parentry['params']

            if parentry['name'] == 'dec_logvar':
                blayercov = False
            else:
                blayercov = self.uqoptions.blayercov

            grad_params = torch.autograd.grad(loss, p, create_graph=True)
            # hv = torch.autograd.grad(g000, p, create_graph=True)
            hess_params = torch.zeros_like(grad_params[0])

            #print hess_params.size()
            bfullcov = False
            bparisvector = False
            dim_grad = grad_params[0].dim()
            if dim_grad == 1:
                bparisvector = True
                if blayercov:
                    bfullcov = True
                    size = grad_params[0].size()
                    elements = size[0]
                    unrolled_grad_params = grad_params[0]

                    hess_params = torch.autograd.Variable(torch.torch.zeros(elements, elements))
                    for i in range(elements):
                        # gives the row of the hessian
                        hessrow = torch.autograd.grad(unrolled_grad_params[i], p, retain_graph=True)[0].resize(elements)
                        hess_params[i, :] = hessrow
                else:
                    bfullcov = False
                    for i in range(grad_params[0].size(0)):
                        hess_params[i] = torch.autograd.grad(grad_params[0][i], p, retain_graph=True)[0][i]
            else:
                bparisvector = False
                #if blayercov:
                # TODO sparse storage of matrix needed
                if False:
                    bfullcov = True
                    size = grad_params[0].size()
                    elements = size[0]*size[1]
                    unrolled_grad_params = grad_params[0].resize(elements)

                    hess_params = torch.autograd.Variable(torch.torch.zeros(elements, elements))
                    for i in range(elements):
                        # gives the row of the hessian
                        hessrow = torch.autograd.grad(unrolled_grad_params[i], p, retain_graph=True)[0].resize(elements)
                        hess_params[i, :] = hessrow
                else:
                    bfullcov = False
                    for i in range(grad_params[0].size(0)):
                        for j in range(grad_params[0].size(1)):
                            hess_params[i, j] = torch.autograd.grad(grad_params[0][i][j], p, retain_graph=True)[0][i, j]

            if not bfullcov:
                hess_params[hess_params<3.] = 1.e5
            hess_list.append({'name': parentry['name'], 'id': parentry['id'], 'diaghessian': hess_params.data, 'fullcov': bfullcov, 'parisvector': bparisvector})
            np.savetxt(parentry['name']+'_' + str(self.N) + '.txt', hess_params.data.cpu().numpy())

        self.vaemodel.bgetlogvar = bvartemp

        return pert_param_list, params_dec_copy, hess_list

    ###

    ###

    def train(self):

        #self.vaemodel = torch.load('/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/ganpeptide/results/VARjoint/model.pth')

        self.train_hist = {}
        self.train_hist['Total_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['kl_qp'] = []

        self.train_hist_pred = {}
        self.train_hist_pred['mu'] = []
        self.train_hist_pred['sig'] = []
        self.train_hist_pred['sig_error_norm'] = []
        self.train_hist_pred['mu_error_norm'] = []
        self.train_hist_pred['kl_ref_to_pred'] = []
        self.train_hist_pred['kl_pred_to_ref'] = []

        parlist = self.getweightlist()

        # store intermetdiate steps
        intermediate = False

        # if the model is loaded no need to train it.
        if not self.bloadmodel:
            print('training start!!')
            start_time = time.time()

            for epoch in range(1, self.epoch + 1):

                if epoch < 1500:
                    weight_vae = 1.0
                else:
                    weight_vae = 0.5

                epoch_start_time = time.time()
                #if self.bCombinedWithData:
                self.trainepochCombined(epoch, weight_vae)
                #else:
                #    self.trainepoch(epoch)

                # visualize intermediate latent space
                self.visLatentTraining(epoch)

                #self.refmodel.doStep()
                # test(epoch)

                if not epoch % self.outputfrequ or epoch == 1:
                    samples = self.gen_samples(n_samples=self.n_samples, iter=epoch, bintermediate=True)
                    #if 'ang' not in self.angulardata:
                    #    samples *= 10.
                    np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + '_' + str(epoch) + '.txt',
                               samples)
                    np.savetxt(os.path.join(self.output_dir, 'train_hist.txt'),
                               self.train_hist['Total_loss'])


                # Visualize intermediate steps, i.e. the latent embedding and the ELBO
                if self.bvislatent_training and not epoch % 10:
                    if hasattr(self.vaemodel, 'plotlatentrep'):
                        if self.gpu_mode:
                            self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                                        path=self.output_dir, postfix='_'+str(epoch))
                        else:
                            self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim,
                                                        path=self.output_dir, postfix='_'+str(epoch))

                        utils.loss_plot(self.train_hist,
                                        self.output_dir,
                                        self.model_name + self.predprefix + '_'+str(epoch), bvae=True, bintermediate=True)

                # Make intermediate predictions during the training procedure
                if intermediate and (not epoch % self.outputfrequ or epoch == 1):

                    # save the parameters for this step
                    sstep = '%05d' % epoch
                    self.storeweightlist(parlist=parlist, path=self.output_dir, prefix=sstep)

                    if hasattr(self.vaemodel, 'plotencodedgaussian'):
                        # produce samples from reference model
                        samples_torch = self.refmodel.sample_torch(nsamples=2000)

                        if self.gpu_mode:
                            samples_torch = samples_torch.cuda()

                        zmu, zlogvar = self.vaemodel.encode(samples_torch)

                        if self.gpu_mode:
                            p = self.refmodel.getpdf(samples_torch.cpu().numpy())
                            zmunp = zmu.detach().cpu().numpy()
                            samp = samples_torch.cpu().numpy()
                        else:
                            p = self.refmodel.getpdf(samples_torch.numpy())
                            zmunp = zmu.detach().numpy()
                            samp = samples_torch.numpy()



                        # plot the encoded samples
                        self.vaemodel.plotencodedgaussian(samples=samp, p_val=p, z_rep=zmunp, path=self.output_dir,
                                                          postfix='_' + str(epoch), name_colormap='viridis',
                                                          mcomponents=self.refmodel.getMixtures())
                        samples = self.gen_samples(n_samples=self.n_samples, iter=epoch, bintermediate=True)

                        # get KL-divergence KL( p_ref(x) || \bar p_\theta(x) )
                        logp = self.refmodel.getlogpdfTorch(samples_torch)
                        logbarptheta = self.vaemodel.getlogpdf(samples_torch, self.gpu_mode)
                        if self.gpu_mode:
                            KLdist = - torch.mean(logbarptheta.cuda() - logp.cuda())
                        else:
                            KLdist = - torch.mean(logbarptheta - logp)
                        KLdist = KLdist.data.cpu().numpy()
                        self.train_hist_pred['kl_pred_to_ref'].append(KLdist)

                        x = range(0, epoch + 1, self.outputfrequ)
                        utils.plt_kldiv(x=x, y=self.train_hist_pred['kl_pred_to_ref'], path=self.output_dir,
                                        filename='KL_pred_to_ref.pdf')
                        np.savetxt(os.path.join(self.output_dir, 'train_hist_kl_pred_to_ref.txt'),
                                   self.train_hist_pred['kl_pred_to_ref'])

                        if self.x_dim == 2:

                            mu, sig = self.vaemodel.plotgaussprediction(samples=samples, path=self.output_dir, postfix='_'+str(epoch))#, refsamples=self.refmodel.getRefSamples())
                            mus = np.copy(mu)
                            sigs = np.copy(sig)

                            self.train_hist_pred['mu'].append(mus)
                            self.train_hist_pred['sig'].append(sigs)
                            np.savetxt(os.path.join(self.output_dir, 'train_hist_mu.txt'), self.train_hist_pred['mu'])
                            np.savetxt(os.path.join(self.output_dir, 'train_hist_sig.txt'), self.train_hist_pred['sig'])

                        if not self.refmodel.getIsMixture():

                            mustorch = self.vaemodel.getMean().detach()
                            mus = mustorch.cpu().numpy()
                            sigstorch = self.vaemodel.getCov().detach().view(1, -1).squeeze()
                            sigs = sigstorch.cpu().numpy()
                            mus_ref = self.refmodel.getNPmu()
                            sigs_ref = self.refmodel.getNPsigs()
                            mu_error = mus - mus_ref
                            mu_error_norm = np.linalg.norm(mu_error)
                            sig_error = sigs - sigs_ref
                            sig_error_norm = np.linalg.norm(sig_error)

                            musreftorch = self.refmodel.getmu().detach()
                            sigsreftorch = self.refmodel.getCov().detach().view(1, -1).squeeze()
                            sigsreftorchcopy = sigsreftorch.clone()
                            sigsreftorchcopy[sigsreftorch == 0] = 1

                            mu_error_rel_torch = (mustorch - musreftorch).abs().mul(
                                musreftorch.abs().reciprocal()).sum().mul(1. / (self.x_dim))
                            mu_error_rel = mu_error_rel_torch.cpu().numpy()
                            sig_error_rel_torch = (sigstorch - sigsreftorch).abs().mul(
                                sigsreftorchcopy.abs().reciprocal()).sum().mul(1. / (self.x_dim * self.x_dim))
                            sig_error_rel = sig_error_rel_torch.cpu().numpy()

                            self.train_hist_pred['mu_error_norm'].append([mu_error_norm, mu_error_rel])
                            self.train_hist_pred['sig_error_norm'].append([sig_error_norm, sig_error_rel])

                            covref = self.refmodel.getCovNP()
                            covinvref = self.refmodel.getInvCovNP()
                            muref = self.refmodel.getNPmu()

                            mupred = self.vaemodel.getMeanNP()
                            covpred = self.vaemodel.getCovNP()

                            sref, logdetcovref = np.linalg.slogdet(covref);
                            detcovref = sref * np.exp(logdetcovref)
                            spred, logdetcovpred = np.linalg.slogdet(covpred);
                            detcovpred = spred * np.exp(logdetcovpred)

                            murefmmupred = muref - mupred
                            D = np.inner(np.inner(murefmmupred, covinvref.T), murefmmupred)

                            A = np.log(detcovref / detcovpred)
                            B = self.x_dim
                            C = np.trace(np.inner(covinvref, covpred))

                            kldiv = 0.5 * (A - B + C + D)
                            self.train_hist_pred['kl_ref_to_pred'].append(kldiv)
                        else:
                            self.train_hist_pred['kl_ref_to_pred'].append(0.)
                            # compute the kl divergence between distributions

                    np.savetxt(os.path.join(self.output_dir, 'train_hist_sig_error.txt'), self.train_hist_pred['sig_error_norm'])
                    np.savetxt(os.path.join(self.output_dir, 'train_hist_mu_error.txt'), self.train_hist_pred['mu_error_norm'])
                    np.savetxt(os.path.join(self.output_dir, 'train_hist_kl.txt'), self.train_hist_pred['kl_ref_to_pred'])

                # store training time for one epoch
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            # save the trained model
            torch.save(self.vaemodel, self.output_dir + '/model.pth')

            # store total training data
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                            self.epoch,
                                                                            self.train_hist['total_time'][0]))
            print("Training finish!... save training results")
            #print("Final KLD loss %.3f" % (self.train_hist['kl_qp'][-1]))

            # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
            #                          self.epoch)

            utils.loss_plot(self.train_hist,
                            self.output_dir,
                            self.model_name + self.predprefix, bvae=True)

        else:
            # load the vae model which was trained before for making predictions
            self.vaemodel = torch.load(self.filemodel)

        # generate samples for predictions (MAP estimate)
        self.gen_samples(self.n_samples)

        quit()

        # Count active parameters
        if self.bard:
            nonactiveparams = self.countzeroweights(paramlist=self.ardprior.paramlist, threshold=0.0001)
        else:
            nonactiveparams = self.countzeroweights(paramlist=self.getdecweightlist(), threshold=0.0001)
        nap = np.ones(1)*nonactiveparams
        np.savetxt(self.output_dir + '/nonactiveparams.txt', nap)

        # model has been trained
        self.vaemodel.eval()  # train(mode=False)

        # generate samples for predictions (MAP estimate)
        self.gen_samples(self.n_samples)

        # calculate the hessian if we require the UQ
        if self.uqoptions.bdouq:
            self.samplePosterior(self.uqoptions.npostsamples)


    def vis_phipsilatent(self, path):
        '''
        Visualize the phi and psi landscape for the learned latent representation. This function creates a grid and
        corresponding to the CVs make predictions x.
        :param path: string, provide path where to save the prediction.
        :return:
        '''

        # create grid with numpy
        x = np.linspace(-4., 4., 101)
        y = np.linspace(-4., 4., 101)
        X, Y = np.meshgrid(x, y)
        Xvec = X.flatten()
        Yvec = Y.flatten()

        # convert numpy array to torch
        Xtorch = torch.from_numpy(Xvec).float()
        Xtorch.unsqueeze_(1)
        Ytorch = torch.from_numpy(Yvec).float()
        Ytorch.unsqueeze(1)

        if self.gpu_mode:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1).cuda())
        else:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1))

        # decode the CVs z to x
        torchsamples = self.vaemodel.decode(samples_z)
        # convert to numpy
        samples = torchsamples.data.cpu().numpy()

        # convert the samples if they are in the angular format
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
        np.savetxt(path + '/samples_vis_phipsi' + self.predprefix + '.txt', samplesout)

    def countzeroweights(self, paramlist, threshold=0.0001):
        '''
        This function counts the inactive weights of the decoding neural network.
        :param paramlist: Torch parameter list of parameters which should be considered.
        :param threshold: Threshold, when to count them as zero.
        :return:
        '''
        counter = 0
        for paramitem in paramlist:
            par = paramitem['params']

            abspar = par.abs()
            abspardat = abspar.data
            smaller = abspardat[abspardat < threshold]
            if smaller.dim() > 0:
                counter = counter + int(smaller.size()[0])

        return counter

    def vis_realizations(self):
        '''
        This functin visualizes a specific amount of realizations per prediction of the decoded data points.
        For showing that the variance of p(x|z) captures outer Hydrogen atoms as noise.
        :return:
        '''

        ic = 0
        # load the
        for batch_idx, data in enumerate(self.data_loader):
            if ic == 0:
                data_vis = data

        #x = Variable(self.data_tensor_vis_1527)
        x = Variable(data_vis)
        n_samples = x.shape[0]
        # encode the data set into latent space
        muenc, log = self.vaemodel.encode(x)

        # decode to mean and variance of predictions
        self.vaemodel.bgetlogvar = True
        if self.gpu_mode:
            samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(muenc).gpu()
        else:
            samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(muenc)  # .cpu()

        mu = samplesTorchmu.data.cpu().numpy()
        vartorch = samplesTorchlogvar.exp()
        var = vartorch.data.cpu().numpy()

        # init total amount of sample mtrix
        nsamples_per_dp = 300
        nsamples_per_dp_tot = nsamples_per_dp + 1
        n_samples_tot = (n_samples ) * nsamples_per_dp_tot
        samples_aevb = np.zeros([n_samples_tot, self.x_dim])

        icount = 0
        for n in xrange(n_samples):
            samples_aevb[n * nsamples_per_dp_tot, :] = mu[n, :]

            samples_aevb[n * nsamples_per_dp_tot + 1: (n + 1) * nsamples_per_dp_tot, :] = np.random.multivariate_normal(
                mu[n, :],
                np.diag(
                    var[n, :]/2.5),
                nsamples_per_dp)
        self.vaemodel.bgetlogvar = False

        samplesoutaevb = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
        np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + '_vis_mean_samples_' + '.txt', samplesoutaevb)



    def vis_latentpredictions(self, yb, ny, path):
        '''
        This function predicts atomistic configurations along a provided path in the CV space. Those z are mapped
        to full atomistic configurations. Currently just ny and path are required.
        :param ny: Number of points in y direction.
        :param path: String of path for storing the prediction.
        :return:
        '''

        # this allows to plot the latent representation and augment it with an indicator at the current position in the latent space
        bVisualizeStar = False

        #y = torch.linspace(yb[0], yb[1], ny)

        # y coordinates
        y = torch.linspace(-4, 4, ny)
        #y1 = torch.linspace(-4, 0, ny*3)
        #y2 = torch.linspace(0, 4, ny*2)
        #nges =  5*ny
        #y = torch.cat((y1,y2))

        # x coordinates
        x = torch.zero(ny)

        # check if gpu mode is active
        if self.gpu_mode:
            y = y.cuda()
            x = x.cuda()

        # summarize x and y in torch variable
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)
        samples_z = Variable(torch.cat((y, x), 1))

        # This is for showing a little star at the current position in the latent space.
        # E.g. for visualizing atomistic configurations for given CVs.
        if bVisualizeStar:
            for i in range(0, ny):
                xnp_curr = samples_z[i, 0].data.numpy()
                ynp_curr = samples_z[i, 1].data.numpy()
                if i==0:
                    n = self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=False)
                else:
                    n = self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=True, normaltemp=n)

        torchsamples = self.vaemodel.decode(samples_z)
        samples = torchsamples.data.cpu().numpy()

        # convert the samples if they are in the angular format
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
        np.savetxt(path + '/samples_vis_latent' + self.predprefix + '.txt', samplesout)

    def gen_samples(self, n_samples=4000, postsampid=None, iter=-1, bintermediate=False):

        # saving samples with postfix
        if postsampid == None:
            postsamplepostfix = ''
        else:
            postsamplepostfix = '_postS_' + str(postsampid)

        # convert the samples if GPU mode is active
        if self.gpu_mode:
            sample_z_ = Variable(
                torch.randn((n_samples, self.z_dim)).cuda())
            z_init = Variable(
                torch.randn((1, self.z_dim)).cuda())
        else:
            sample_z_ = Variable(torch.randn((n_samples, self.z_dim)))
            z_init = Variable(torch.randn((1, self.z_dim)))

        # Utilize pseudo gibbs algorithm as provided in Mattei, 2018. This algorithm corrects for the approximate
        # posterior
        if self.exactlikeli:
            self.vaemodel.bgetlogvar = True
            initmu, initlogvar = self.vaemodel.decode(z_init)
            initstd = initlogvar.mul(0.5).exp_()
            pinit = torch.distributions.Normal(initmu, initstd)
            x_init = pinit.sample()

            pgibs = PseudoGibbs(x_init, z_init, self.vaemodel)
            samples_aevb_gibbs = pgibs.sample(n_samples * self.n_samples_per_mu)
            samplesnp_aevb_gibbs = samples_aevb_gibbs.data.cpu().numpy()
            if self.autoencvarbayes:
                samplesoutaevbgibbs = convert_given_representation(samples=samplesnp_aevb_gibbs,
                                                                   coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
                if not bintermediate:
                    np.savetxt(self.output_dir + '/samples_aevb_gibbs' + self.predprefix + postsamplepostfix + '.txt',
                           samplesoutaevbgibbs)
                del samplesoutaevbgibbs
            self.vaemodel.bgetlogvar = False

        # Prediction for Variational Autoencoder. I.e. sample p(z), and project to x directly with \mu(z).
        # No probabilistic model employed for the mapping in this case.
        if not self.autoencvarbayes:
            if self.gpu_mode:
                samplesTorchmu = self.vaemodel.decode(sample_z_)#.gpu()
            else:
                samplesTorchmu = self.vaemodel.decode(sample_z_)  # .cpu()

            # samples are the means directly
            samples = samplesTorchmu.data.cpu().numpy()
            # convert the samples if they are in the angular format
            samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
            if not bintermediate:
                np.savetxt(self.output_dir + '/samples' + self.predprefix + postsamplepostfix + '.txt', samplesout)

        # Prediction for Auto-Encoding Variational Bayes. I.e. sample p(z),
        # given those SAMPLE p(x|z) = N(\mu, \sigma^2). This corresponds to the actual AEVB algorithm and is Bayesian.
        else:
            # Provide at least one sample per mean prediction.
            if self.n_samples_per_mu == 0:
                aevb_samples_per_mu = 1
            else:
                aevb_samples_per_mu = self.n_samples_per_mu

            # in case of AEVB one requires the mean and variance of the predictive model p(x|z). Enable this here.
            self.vaemodel.bgetlogvar = True
            if self.gpu_mode:
                samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(sample_z_) #.cuda()
            else:
                samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(sample_z_)  # .cpu()

            # TODO do not convert here to numpy for sampling from gaussian but use instead the torch implementation
            # of the Normal distribution. Those should be inculded in pyTorch 0.4.0
            mu = samplesTorchmu.data.cpu().numpy()
            vartorch = samplesTorchlogvar.exp()
            var = vartorch.data.cpu().numpy()

            if self.bDebug:
                np.savetxt(os.path.join(self.output_dir, 'var_pred.txt'), var)

            # init total amount of sample matrix
            n_samples_tot = n_samples * aevb_samples_per_mu
            samples_aevb = np.zeros([n_samples_tot, self.x_dim])

            icount = 0
            # sample the p(x|z) for different CVs z and its corresponding \mu(z), \sigma(z).
            for n in xrange(n_samples):
                samples_aevb[n * aevb_samples_per_mu:(n + 1) * aevb_samples_per_mu, :] = np.random.multivariate_normal(
                    mu[n, :],
                    np.diag(
                        var[n, :]),
                    aevb_samples_per_mu)
            self.vaemodel.bgetlogvar = False

            # store the predictions
            samplesout = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata, unitgiven=self.coordinatesiunit)
            if not bintermediate:
                np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + postsamplepostfix + '.txt',
                       samplesout)
        return samplesout

    def visLatentTraining(self, epoch):
        # Visualize intermediate steps, i.e. the latent embedding and the ELBO
        if self.bvislatent_training and not epoch % 50:
            if hasattr(self.vaemodel, 'plotlatentrep'):
                if self.gpu_mode:
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide)
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_dat_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide,  x_train=Variable(self.data_tensor).cuda())
                else:
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide)
                    self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim,
                                                path=self.output_dir, postfix='_dat_' + str(epoch), data_dir=self.data_dir, peptide=self.name_peptide, x_train=Variable(self.data_tensor))

                utils.loss_plot(self.train_hist,
                                self.output_dir,
                                self.model_name + self.predprefix + '_' + str(epoch), bvae=True, bintermediate=True)

    def postprocessing(self, n_samples=4000, postsampid=None):
        '''
        This function provides predictions given the trained model. In the case of \dim(z) = 2,
        further visualizations are issued automatically.
        :param n_samples: Amount of requred samples of z \sim p(z)
        :param postsampid: Do no specify this, it is just required internally for sampling the posterior of the decoding
        parametrization.
        :return:
        '''

        # visualize latent representation if z_dim = 2
        if self.z_dim == 2 and postsampid == None and self.name_model == 'ala_2':
            yborder = np.array([4., -4.])
            # create predictions for the latent representation
            self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
            # visualize the phi-psi landscape given the CVs
            self.vis_phipsilatent(path=self.output_dir)

        # visualize mapping between the different layers
        # TODO Add this for the vae model
        if hasattr(self.vaemodel, 'plotdecoder') and postsampid == None and self.name_model == 'ala_2':
            self.vaemodel.plotdecoder(n_samples=500, z_dim=self.z_dim)
        else:
            print 'No visualization for decoder available.'

        # visualize the mapping from input to latent space
        if hasattr(self.vaemodel, 'plotencoder') and postsampid == None:
            data_loader_visualization = DataLoader(TensorDatasetDataOnly(self.data_tensor),
                                                   batch_size=1527,
                                                   shuffle=False, **self.kwargsdatloader)

            for index, data in enumerate(data_loader_visualization):
                data = Variable(data)
                if self.gpu_mode:
                    data = data.cuda()
                self.vaemodel.plotencoder(x=data, z_dim=self.z_dim, strindex=str(index))
        elif postsampid == None:
            print 'No visualization for encoder available.'

        if hasattr(self.vaemodel, 'plotlatentrep') and postsampid == None:
            if self.gpu_mode:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                            path=self.output_dir)
            else:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir)

        # visualize realizations along the z_1 or z_2 axis to show that the variance captures
        if False:
            self.vis_realizations()

#        if postsampid == None:
#            if not self.bClusterND:
#                print(np.amax(samplesout))
#                print(np.amin(samplesout))
#                print(np.mean(samplesout))
#                print(np.std(samplesout))
#                print('Done generating {} samples'.format(self.n_samples))

#                real_data = np.loadtxt('../../data_peptide/dataset_alpha_10000_sub_1000.txt').T
#                print(np.amax(real_data))
#                print(np.amin(real_data))
#                print(np.mean(real_data))
#                print(np.std(real_data))
