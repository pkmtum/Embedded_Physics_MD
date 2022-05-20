import utils, torch, time, os, pickle, datetime

# import torch._utils
# try:
#     torch._utils._rebuild_tensor_v2
# except AttributeError:
#     def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#         tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#         tensor.requires_grad = requires_grad
#         tensor._backward_hooks = backward_hooks
#         return tensor
#     torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import warnings
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
from utils_peptide import convert_given_representation
from utils_peptide_torch import convertReferenceDataToUnit
from utils_peptide_torch import convertToReducedCartersianCoordinates
from utils_peptide_torch import convertToFullCartersianCoordinates
from utils_peptide import estimateProperties

from utils import write_command

# Import classes related to MD simulations
import math
from MDLoss import MDLoss
from MDLoss import MDSimulator
from MDLoss import MDImportanceSampler
from GaussianRefModelParametrization import GaussianRefModelParametrization as GaussRefParams

class ARDprior:
    def __init__(self, a0, paramlist, model, gpu):
        self.model = model
        self.paramlist = paramlist
        self.a0 = a0
        self.b0 = a0
        self.gpu = gpu

    def getlogpiorARD(self):
        sumlogptheta = 0.
        for paramitem in self.paramlist:
            par = paramitem['params']

            psqu = par.pow(2.)
            with torch.no_grad():
                denominator = psqu.mul(0.5) + self.b0
                nominator = torch.zeros_like(denominator)
                nominator.fill_(self.a0 + 0.5)
                expectau = nominator.div(denominator)
            logptheta = expectau.mul(psqu)
            logptheta.mul_(-0.5)
            sumlogptheta += logptheta.sum()

        return sumlogptheta


class PseudoGibbs:
    def __init__(self, x_init, z_init, model):
        self.x_init = x_init
        self.z_init = z_init
        self.model = model
        self.model.bgetlogvar = True

        self.n_skip = 10
        self.n_init = 5000

    def sampleposterior(self, x):
        mu, logvar = self.model.encode(x)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        sample = post.sample()
        return sample

    def samplepredictive(self, z):
        self.model.bgetlogvar = True
        mu, logvar = self.model.decode(z)
        std = logvar.mul(0.5).exp_()
        pred = torch.distributions.Normal(mu, std)
        x = pred.sample()
        return x

        mu, logvar = self.model.encode(x)
        post = torch.distributions.Normal(mu, logvar.exp_())
        sample = post.sample()
        return sample

    def evallogprobposterior(self, x, z):
        mu, logvar = self.model.encode(x)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        logprob = post.log_prob(z).sum()
        return logprob

    def evallogprobcondpred(self, x, z):
        self.model.bgetlogvar = True
        mu, logvar = self.model.decode(z)
        std = logvar.mul(0.5).exp_()
        post = torch.distributions.Normal(mu, std)
        logprob = post.log_prob(x).sum()
        return logprob

    def evallogprobprior(self, z):
        mu = torch.zeros_like(z)
        scale = torch.ones_like(z)
        prior = torch.distributions.Normal(mu, scale)
        logprob = prior.log_prob(z).sum()
        return logprob

    def calcacceptanceratio(self, ztm1, ztprop, xtm1):

        p_xtm1_given_ztprop = self.evallogprobcondpred(xtm1, ztprop)
        p_xtm1_given_ztm1 = self.evallogprobcondpred(xtm1, ztm1)

        p_ztprop = self.evallogprobprior(ztprop)
        p_ztm1 = self.evallogprobprior(ztm1)

        q_ztm1_given_xtm1 = self.evallogprobposterior(xtm1, ztm1)
        q_ztprop_given_xtm1 = self.evallogprobposterior(xtm1, ztprop)

        ratio_pxgz = p_xtm1_given_ztprop - p_xtm1_given_ztm1
        ratio_pz = p_ztprop - p_ztm1
        ratio_qzgx = q_ztm1_given_xtm1 - q_ztprop_given_xtm1

        logroh =  ratio_pxgz + ratio_pz + ratio_qzgx

        return logroh.exp()

    def getboolofvariable(self, bytetensor):
        res = bool(bytetensor[0])
        return res

    def sample(self, N):
        '''
        Obtain samples for
        :param N:
        :return:
        '''

        n_tot = self.n_init + N*self.n_skip
        n_accepted = 0

        xtm1 = self.x_init
        ztm1 = self.z_init

        x_samples = xtm1

        for i in range(n_tot):
            ztprop = self.sampleposterior(xtm1)
            rhot = self.calcacceptanceratio(ztm1, ztprop, xtm1)
            rhottensor = rhot.data[0]

            if rhottensor > 1.:
                zt = ztprop
                n_accepted += 1
            else:
                r = torch.rand(1)
                if rhottensor > r[0]:
                    zt = ztprop
                    n_accepted += 1
                else:
                    zt = ztm1

            xt = self.samplepredictive(zt)

            s = i - self.n_init
            if s > 1 and s % self.n_skip == 0:
                x_samples = torch.cat((x_samples, xt), 0)

            ztm1 = zt
            xtm1 = xt

        accept_ratio = n_accepted/float(n_tot)
        print(accept_ratio)

        return x_samples


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

class VARjoint(object):
    def __init__(self, args, system_arguments=None):
        # parameters
        self.system_args = system_arguments

        self.abs_kl_increase_meassure = False

        self.output_en_decoded = False
        self.epoch = args.epoch
        self.sample_num = 64
        self.batch_size = args.batch_size
        # self.batch_size = 64
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = bool(args.gpu_mode) and torch.cuda.is_available()
        if self.gpu_mode:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.dropout_enc_dec = args.dropout_enc_dec
        self.dropout_p = args.dropout

        self.model_name = args.model_type
        self.c = args.clipping  # clipping value
        self.n_critic = args.n_critic  # the number of iterations of the critic per generator iteration
        self.z_dim = args.z_dim
        self.n_samples = args.samples_pred

        self.cluster_name = args.cluster
        if self.cluster_name == 'ND':
            self.bClusterND = True
            self.bClusterTUM = False
        elif self.cluster_name == 'TUM':
            self.bClusterND = False
            self.bClusterTUM = True
        else:
            self.bClusterND = False
            self.bClusterTUM = False

        # check if any cluster options do not fit
        if hasattr(args, 'clusterND'):
            bClusterND_temp = bool(args.clusterND)
            if bClusterND_temp != self.bClusterND:
                raise ValueError(
                    'Please check the deprecated option \'--clusterND\'.This should coincide with \'--cluster\'')

        self.output_postfix = args.outPostFix
        self.angulardata = args.useangulardat
        self.autoencvarbayes = bool(args.AEVB)
        self.L = args.L  # amount of eps ~ p(eps) = N(0,1)
        self.Z = args.Z  # amount of samples from p(z)
        self.outputfrequ = args.outputfreq
        self.n_samples_per_mu = args.samples_per_mean  # if 0, just use mean prediction: x = mu(z)
        self.lambdaexpprior = args.exppriorvar

        self.exactlikeli = bool(args.exactlikeli)

        self.convolve_pot_sig = args.convolute_target_potential_sig
        self.convolve_pot_n = args.convolute_target_potential_n

        # gradient of MD code postprocessing
        self.md_grad_postproc = args.md_grad_postproc

        self.add_gauss_ref = bool(args.add_reference_gaussian)
        if args.add_gaussian_sig_sq <= 0.:
            self.gauss_ref_sig_sq = None
        else:
            self.gauss_ref_sig_sq = args.add_gaussian_sig_sq

        bqu = bool(args.npostS)
        self.uqoptions = UQ(bdouq=bqu, bcalchess=True, blayercov=False, buqbias=bool(args.uqbias))
        self.uqoptions.npostsamples = args.npostS
        self.bfixlogvar = bool(args.sharedlogvar)
        self.bfixenclogvar = bool(args.sharedencoderlogvar)
        self.betaVAE = args.betaVAE

        # check if a trained model should be loaded
        self.filemodel = args.loadtrainedmodel
        self.bloadmodel = bool(self.filemodel)
        self.bloadstatedict = bool(args.loadstatedict)
        if self.bloadmodel and self.bloadstatedict:
            raise ValueError('Either load a model or load a state_dict from a model and continue training.')
        elif self.bloadmodel:
            self.filemodel = args.loadtrainedmodel
        elif self.bloadstatedict:
            self.filemodel = args.loadstatedict
            try:
                temp_model = torch.load(self.filemodel)
            except OSError:
                raise IOError('File is not readable.')
            except:
                temp_model = torch.load(self.filemodel,  map_location='cpu')
            import copy
            self.loaded_state_dict = copy.deepcopy(temp_model.state_dict())

            # free the memory for this object
            del temp_model
        else:
            self.filemodel = ''

        self.bvislatent_training = True
        self.bvismean_and_samples = False

        self.bassigrandW = bool(args.assignrandW)
        self.bfreememory = bool(args.freeMemory)

        # step scheduler options
        if args.stepSched > 50:
            imaxstepssched = args.stepSched
        else:
            imaxstepssched = 800
        self.stepschedopt = {'usestepsched': bool(args.stepSched), 'stepschedresetopt': bool(args.stepSchedresetopt),
                             'imaxstepssched': imaxstepssched, 'stepschedconvcrit': args.stepSchedintwidth,
                             'stepschedtype': args.stepSchedType, 'sched_method_check_convergence': args.convergence_criterion}
        self.betaPrefactor_init = args.setBetaPrefactor

        self.bseplearningrate = bool(args.separateLearningRate)
        if not (self.bfixlogvar or self.bfixenclogvar):
            self.bseplearningrate = False
        self.breddescr = bool(args.redDescription)
        if ('ang' in self.angulardata) and self.breddescr:
            raise ValueError('Error: Reduced description is not implemented for angular representation.')

        # select the forward model
        self.x_dim = args.x_dim
        self.x_dim_mod = args.x_dim
        self.joint_dim = self.x_dim + self.z_dim

        self.coordinatesiunit = 1.e-9
        self.coorddataprovided = 1.e-10
        self.bDebug = True
        self.bCombinedWithData = False
        self.weightvaeinit = 0.999

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

        # networks init
        if self.angulardata == 'ang':
            self.x_dim = (22 - 1) * 3
            print('Error: Model for %s not implemented yet.' % self.angulardata)
            quit()
            # from NET_ang import generator
            # from NET_ang import discriminator
        elif self.angulardata == 'ang_augmented':
            self.x_dim = (22 - 1) * 5
            print('Error: Model for %s not implemented yet.' % self.angulardata)
            quit()
            # from NET_angaug import generator
            # from NET_angaug import discriminator
        elif self.angulardata == 'ang_auggrouped':
            self.x_dim = (22 - 1) * 5

            if self.autoencvarbayes:
                from VAEmodelKingma import VAEmodauggrouped as VAEmod
                # from VAEmodelKingma import VAEmodangauggroupedsimple as VAEmod
            else:
                from VAEmodel import VAEmodangauggroupedsimple as VAEmod

            # from VAEmodel import VAEmodangauggroupedlong as VAEmod

            # from NET_angaug_long import generator
            # from NET_angaug import discriminator
        else:
            # TODO move all that to a loader class
            if 'ala_15' in self.dataset:
                self.x_dim = 162 * 3
                self.x_dim_mod = self.x_dim
            elif 'gauss' in self.dataset:
                self.x_dim_mod = self.x_dim
            # this is for ala-2
            elif 'quad' in self.dataset:
                self.x_dim = 2
                self.x_dim_mod = 2
            else:
                self.x_dim = 66
                self.x_dim_mod = self.x_dim
                if self.breddescr:
                    self.x_dim_mod = 60

            if self.autoencvarbayes:
                from VAEmodelKingma import VAEmod
            else:
                from VAEmodel import VAEmodsimple as VAEmod
                # from VAEmodel import VAEmodcoordlong as VAEmod
            # from VAEmodel import VAEmodcoordlong as VAEmod

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
        elif self.bClusterTUM:
            data_dir = '/home/markus/projects/data/data_peptide'
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
                data_dir + '/dataset_1500' + angpostfix + '.txt').T)
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

        # convert the data_tensor into the correct and consistent units
        self.unitlenghfactor = self.coorddataprovided / self.coordinatesiunit
        #self.unitlenghfactor = 1.0

        #if self.gpu_mode:
        #    data_tensor = data_tensor.to('cuda')

        if self.dataset not in ['var_gauss', 'quad']:
            convertReferenceDataToUnit(data_tensor, self.unitlenghfactor, self.angulardata)
            data_tensor = convertToReducedCartersianCoordinates(data=data_tensor, breduce=self.breddescr)

        # categorize what to do. Combine data and reverse variational approach or not
        if 'ala_15' in self.name_peptide:
            print('We do not support ALA15 peptide in the current version.')
            quit()
        elif (self.dataset in ['var_gauss', 'quad', 'ala_2']):
            self.bCombinedWithData = False
            self.N = 0
            # specify the model name and prefix
            if self.dataset == 'var_gauss':
                self.name_model = 'gauss'
                predictprefix = '_gauss'
                self.x_dim = 2
            elif self.dataset == 'quad':
                self.name_model = 'quad'
                predictprefix = '_quad'
                self.x_dim = 2
            elif self.dataset == 'ala_2':
                self.name_model = 'ala_2'
                predictprefix = '_ala_2'
        # in this case we combine the ala_2 reverse variational model with VAE
        else:
            self.bCombinedWithData = True
            self.name_model = 'ala_2'


        if self.dataset not in ['var_gauss', 'quad']:
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
                self.data_tensor_vis_1527 = torch.Tensor(np.loadtxt(data_dir + '/dataset_mixed_1527' + angpostfix + '.txt').T)
            elif 'ala_15' in self.dataset:
                self.data_tensor_vis_1527 = torch.Tensor(np.loadtxt(data_dir + '/ala-15/dataset_ala_15_1500' + angpostfix + '.txt').T)

            #if self.gpu_mode:
            #    self.data_tensor_vis_1527 = self.data_tensor_vis_1527.to('cuda')

            convertReferenceDataToUnit(self.data_tensor_vis_1527, self.unitlenghfactor, self.angulardata)
            self.data_tensor_vis_1527 = convertToReducedCartersianCoordinates(data=self.data_tensor_vis_1527, breduce=self.breddescr)

        # for test purposes add the following
        # train_loader = torch.utils.data.DataLoader(
        #    datasets.MNIST('../data', train=True, download=True,
        #                   transform=transforms.ToTensor()),
        #    batch_size=args.batch_size, shuffle=True, **kwargs)

        # specify as model_name the general kind of dataset: mixed or separate
        self.postfix_setting = self.dataset + '_z_' + str(self.z_dim) + '_b_' + str(
            self.batch_size) + '_svar_' + str(int(self.bfixenclogvar)) + str(
            int(self.bfixlogvar)) + '_lagphi_' + str(args.laggingInferenceStepsPhi) + '_lagtheta_' + str(
            args.laggingInferenceStepsTheta) + '_norm_' + args.md_grad_postproc + '_sched_' + str(args.stepSched)
        self.predprefix = predictprefix

        # saving directory
        working_dir = os.getcwd()
        tempdir = os.path.join(working_dir, self.result_dir, self.model_name, foldername, self.output_postfix,
                               args.output_path, self.postfix_setting)
        self.output_dir = checkandcreatefolder(dir=tempdir)

        # Write the command the run was started with.
        write_command(os.path.join(self.output_dir, 'command.sh'), system_arguments)

        # TODO add this selection task in class or function
        # import the reference model
        nModes = 2
        if self.name_model not in ['gauss', 'quad']:
            from MDLoss import MDSimulator as ReferenceModel
            self.MDLossapplied = MDLoss.apply

            if 'ang' in self.angulardata:
                from VARJmodel import VARmdAngAugGrouped as VARmod
            else:
                from VARJmodel import VARmd as VARmod
                #from VARJmodel import VARmdSepDec as VARmod
                #from VAEmodelKingma import VAEmod as VARmod
        else:
            if self.name_model in 'gauss':
                if nModes == 1:
                    from RefModelGauss import ReferenceModelSingleGauss as ReferenceModel
                    from VARJmodel import VARmod as VARmod
                else:
                    from RefModelGauss import ReferenceModelMultiModal as ReferenceModel
                    #from VARJmodel import VARmixture as VARmod
                    from VARJmodel import VARmixturecomplex as VARmod
                    #from VARJmodel import VARmixturecomplexDeep as VARmod
            elif self.name_model in 'quad':
                from RefModelGauss import ReferenceModelEnergyFunctional as ReferenceModel
                # from VARJmodel import VARmixture as VARmod
                from VARJmodel import VARmixturecomplex as VARmod
                # from VARJmodel import VARmixturecomplexDeep as VARmod

        # initialize the reference model
        if self.name_model not in ['gauss', 'quad']:
            if self.bClusterND:
                reffolderPDB = '/afs/crc.nd.edu/user/m/mschoebe/Private/data/data_peptide/filesALA2/reftraj/'
            elif self.bClusterTUM:
                reffolderPDB = '/home/markus/projects/data/data_peptide/filesALA2/reftraj/'
            else:
                reffolderPDB = '/home/schoeberl/Dropbox/PhD/projects/2018_07_06_openmm/ala2/'

            if self.add_gauss_ref:
                ref_config = self.data_tensor_vis_1527[0, :]
            else:
                ref_config = None
            self.refmodel = MDSimulator(pdbstructure=os.path.join(reffolderPDB, 'ala2_adopted.pdb'), bGPU=self.gpu_mode,
                                        sAngularRep=self.angulardata, sOutputpath=self.output_dir,
                                        stepschedopt=self.stepschedopt, breddescr=self.breddescr,
                                        gradientpostproc=self.md_grad_postproc,
                                        reference_configuration=ref_config,
                                        gaussian_sig_sq=self.gauss_ref_sig_sq, convolve_pot_sig=self.convolve_pot_sig,
                                        convolve_pot_n=self.convolve_pot_n, a_init_preset=self.betaPrefactor_init)
        # specify reference model (only needed if not MD run)
        else:
            if self.name_model in 'gauss':
                muref, sigmaref, W_ref = GaussRefParams.getParVectors(x_dim=self.x_dim, z_dim=self.z_dim, nModes=nModes,
                                                                 bassigrandW=self.bassigrandW)
                self.refmodel = ReferenceModel(mu=muref, sigma=sigmaref, W=W_ref, outputdir=self.output_dir,
                                          bgpu=self.gpu_mode, stepschedopt=self.stepschedopt)
            elif self.name_model in 'quad':
                self.refmodel = ReferenceModel(outputdir=self.output_dir, bgpu=self.gpu_mode,
                                               stepschedopt=self.stepschedopt)
            warnings.warn(
                'For model {}, the absolute KL increase for scheduling is not available.'.format(self.model_name))
            self.abs_kl_increase_meassure = False
            self.refmodel.plot(path=self.output_dir)

        ###################################################################
        self.vaemodel = VARmod(args=args, x_dim=self.x_dim_mod, bfixlogvar=self.bfixlogvar,
                               bfixenclogvar=self.bfixenclogvar, device=self.device, dropout_p=self.dropout_p,
                               dropout_enc_dec=self.dropout_enc_dec)
        ###################################################################
        # check if a state dict should be loaded
        if self.bloadstatedict:
            # TODO map model if it was computed on gpu or cpu accordingly
            self.vaemodel.load_state_dict(self.loaded_state_dict)

        if args.laggingInferenceStepsTheta > 0 or args.laggingInferenceStepsPhi > 0:
            from LaggingInference import LaggingInference as LagInf
            self.laginstance = LagInf(modelobject=self, steps_theta=args.laggingInferenceStepsTheta,
                                 steps_phi=args.laggingInferenceStepsPhi)

        # load importance sampler for approximating the
        if self.name_model not in ['gauss', 'quad'] and self.abs_kl_increase_meassure:
            self.md_importance_sampler = MDImportanceSampler(mdloss=self.MDLossapplied, mdrefmodel=self.refmodel,
                                                        q_dist=self.vaemodel, dim=self.x_dim_mod, dim_z=self.z_dim,
                                                        device=self.device)

        if self.gpu_mode:
            self.vaemodel.cuda()

        # set the optimizer
        self.setOptimizer()
        #self.optimizer = self.setOptimizer()

        if self.bard:
            #self.ardprior = ARDprior(self.arda0, self.getdecweightlist(), self.vaemodel, self.gpu_mode)
            weight_list_for_ard_dec = self.getweightlist_containing_input_list(['dec_'])
            weight_list_for_ard_enc = self.getweightlist_containing_input_list(['enc_'])
            #weight_list_for_ard = self.getdecweightlist()
            self.ardprior_dec = ARDprior(self.arda0, weight_list_for_ard_dec, self.vaemodel, self.gpu_mode)
            self.ardprior_enc = ARDprior(self.arda0, weight_list_for_ard_enc, self.vaemodel, self.gpu_mode)

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
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if 'dec_' in name:
                    # check if we want to uq bias uncertainty
                    if not ('.bias' in name) and not ('dec_logvar' in name):
                        decoding_weight_list.append({'name': name, 'id': id, 'params': param})
                        #pclone = param.clone()
                        #params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print(name)  # , param.data
                    id = id + 1
        return decoding_weight_list

    def getweightlist_containing_input_list(self, param_contains):
        weight_list = []
        id = 0
        for name, param in self.vaemodel.named_parameters():
            if param.requires_grad:
                # UQ only for decoding network
                if any([s in name for s in param_contains]):
                    # check if we want to uq bias uncertainty
                    if not ('.bias' in name) and not ('logvar' in name):
                        weight_list.append({'name': name, 'id': id, 'params': param})
                        print(name)  # , param.data
                    id = id + 1
        return weight_list

    def obtain_indiv_sample_variationalmodel(self):
        N_z = self.batch_size
        N_xpz = self.L
        data_z = torch.randn((N_z, self.z_dim), device=self.device)
        data_z_aug = data_z.repeat(N_xpz, 1)

        # ,mu and logvar of p(x|z)
        data_x_rev_variational, q_recon_batch, pmu, plogvar = self.vaemodel.forward(data_z_aug)
        qmu = q_recon_batch[0]
        qlogvar = q_recon_batch[1]

        return qmu, qlogvar, data_x_rev_variational, data_z_aug, pmu, plogvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_variationalmodel(self, qmu, qlogvar, x_data, z_data, pmu, plogvar, N_z, N_zpx, x_dim=784, bgpu=False, normalize=False, bpxOnly=False, beta_prefactor=None):
        analytic_entropy = False
        bcov = True
        individual_samples = False

        log2pi = torch.tensor(2 * math.pi, device=self.device).log()

        if individual_samples:
            qmu, qlogvar, x_data, z_data, pmu, plogvar = self.obtain_indiv_sample_variationalmodel()

        # < < log p(x) >_p(x|z) >_p(z)
        if self.refmodel.getModelType() == 'GaussianMixture':
            logpx = self.refmodel.logpx(x=x_data, bgpu=bgpu, bfreememory=self.bfreememory)
        elif self.refmodel.getModelType() == 'MD':
            logpx = self.MDLossapplied(x_data, self.refmodel, True, beta_prefactor)
        elif self.refmodel.getModelType() == 'Gaussian':
            logpx = self.refmodel.logpx(x=x_data, bgpu=bgpu, bfreememory=self.bfreememory)
        elif self.refmodel.getModelType() == 'Quad':
            logpx = self.refmodel.logpx(x=x_data, beta=beta_prefactor)
        else:
            print('Not implemented so far. Try another time.')
            quit()


        if individual_samples:
            qmu, qlogvar, x_data, z_data, pmu, plogvar = self.obtain_indiv_sample_variationalmodel()

        # < < log q(z|x) >_p(x|z) >_p(z)
        pointwiseLogqzgx = -0.5 * F.mse_loss(z_data, qmu, reduction='none')# size_average=False, reduce=False)
        # TODO Remove this
        #weightqzgx_temp = qlogvar.exp().reciprocal() # sigsq.reciprocal()  # 1./sigsq
        #pointwiseWeightedMSElossLogqzgx_temp = pointwiseLogqzgx.mul(weightqzgx_temp)

        sigsqqzgx = qlogvar.exp()
        pointwiseWeightedMSElossLogqzgx = pointwiseLogqzgx.div(sigsqqzgx)

        logqzgx = pointwiseWeightedMSElossLogqzgx.sum()
        if self.bfreememory:
            del pointwiseWeightedMSElossLogqzgx, pointwiseLogqzgx

        logqzgx -= 0.5 * qlogvar.sum()
        logqzgx -= self.z_dim * 0.5 * log2pi * N_z * N_zpx
        #logqzgx -= 0.5 * (torch.ones_like(z_data)).mul(2 * math.pi)


        if individual_samples:
            qmu, qlogvar, x_data, z_data, pmu, plogvar = self.obtain_indiv_sample_variationalmodel()

        # < < log p(x|z) >_p(x|z) >_p(z)
        if analytic_entropy:
            entropy_qxgz = 0.5 * plogvar.sum()
            entropy_qxgz += self.x_dim_mod * 0.5 * log2pi * N_z * N_zpx
            entropy_qxgz += self.x_dim_mod * 0.5 * N_z * N_zpx
            logpxgz = -entropy_qxgz
        else:
            pointwiseLogpxgz = -0.5 * F.mse_loss(x_data, pmu, reduction='none')#size_average=False, reduce=False)
            # TODO Remove this
            #weightpxgz_temp = plogvar.exp().reciprocal() # sigsq.reciprocal()  # 1./sigsq
            #pointwiseWeightedMSElossLogpxgz_temp = pointwiseLogpxgz.mul(weightpxgz_temp)
            sigsqpxgz = plogvar.exp()
            pointwiseWeightedMSElossLogpxgz = pointwiseLogpxgz.div(sigsqpxgz)

            logpxgz = pointwiseWeightedMSElossLogpxgz.sum()
            if self.bfreememory:
                del pointwiseWeightedMSElossLogpxgz, pointwiseLogpxgz
            logpxgz -= 0.5 * plogvar.sum()
            logpxgz -= self.x_dim_mod * 0.5 * log2pi * N_z * N_zpx

        # < < log p(z) >_p(x|z) >_p(z)

        if analytic_entropy:
            entropy_qz = (self.z_dim * 0.5 + self.z_dim * 0.5 * log2pi) * N_z * N_zpx
            logpz = -entropy_qz
        else:
            pointwiseLogpz = -0.5 * F.mse_loss(z_data, torch.zeros_like(z_data), reduction='none')#size_average=False, reduce=False)
            weightpz = torch.ones_like(z_data)
            pointwiseWeightedMSElossLogpz = pointwiseLogpz.mul(weightpz)
            logpz = pointwiseWeightedMSElossLogpz.sum()
            logpz -= self.z_dim * 0.5 * log2pi * N_z * N_zpx

        nancheck = torch.tensor([logqzgx, logpx, logpxgz, logpz])

        nans = torch.isnan(nancheck)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            print(nancheck)

        if bpxOnly:
            loss = - logpx
        else:
            if self.betaVAE == 1.:
                loss = - logqzgx - logpx + logpxgz + logpz
            else:
                loss = - logqzgx - logpx + self.betaVAE * logpxgz + self.betaVAE * logpz

        if self.bard:
            ardcontrib_dec = self.ardprior_dec.getlogpiorARD()
            #ardcontrib.div_(float(N_z * N_zpx))
            loss.add_(ardcontrib_dec)

            ardcontrib_enc = self.ardprior_enc.getlogpiorARD()
            #ardcontrib.div_(float(N_z * N_zpx))
            loss.add_(-ardcontrib_enc)

        if normalize:
            loss.div_(float(N_z * N_zpx))

        #if loss.item() > 14851.:
        #    print('Large loss!')

        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, x_dim=784):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False, reduce=False)
        # TODO Check 0.5 factor - should arise by Guassian: -1/2(x-mu)^T \Sigma^-1 (x-mu)
        BCE = F.mse_loss(recon_x, x.view(-1, x_dim), size_average=False, reduce=False)

        # print(logvar.exp())

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.train_hist['kl_qp'].append(KLD)

        return BCE + KLD

    # Reconstruction + KL divergence losses summed over all elements and batch
    # TODO Extend this with a full covariance matrix.
    def loss_function_autoencvarbayesCholVar(self, recon_mu, recon_logvar, x, mu, logvar, x_dim=784):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False, reduce=False)
        pointwiseMSEloss = 0.5 * F.mse_loss(recon_mu, x.view(-1, x_dim), size_average=False, reduce=False)
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

        self.train_hist['kl_qp'].append(KLD)

        # Prior on predictive variance
        psigsqlamb = 30.0
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

        # return (WeightedMSEloss + KLD)
        loss = (logvarobjective + WeightedMSEloss + KLD + logpriorvar)
        lossnp = loss.data.cpu().numpy()
        if lossnp != lossnp:
            print('Error: Loss is NaN')
        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_autoencvarbayes(self, recon_mu, recon_logvar, x, mu, logvar, x_dim=784, normalize=False):

        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False, reduce=False)
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

        if normalize:
            loss.div_(Maug)

        lossnp = loss.data.cpu().numpy()
        if lossnp != lossnp:
            print('Error: Loss is NaN')
        return loss

    # Reconstruction + KL divergence losses summed over all elements and batch
    def dloss_function_autoencvarbayes(self, recon_mu, recon_logvar, x, mu, logvar, x_dim=784):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, x_dim), size_average=False, reduce=False)
        pointwiseMSEloss = 0.5 * F.mse_loss(recon_mu, x.view(-1, x_dim), size_average=False, reduce=False)
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
        loss = (logvarobjective + WeightedMSEloss + KLD + logpriorvar)
        lossnp = loss.data.cpu().numpy()
        if lossnp != lossnp:
            print('Error: Loss is NaN')
        return loss

    def assess_loss(self, beta_prefactor=None):
        self.vaemodel.eval()
        N_z = self.batch_size
        N_xpz = self.L
        data_z = torch.randn((N_z, self.z_dim), device=self.device)
        data_z_aug = data_z.repeat(N_xpz, 1)

        # ,mu and logvar of p(x|z)
        data_x_rev_variational, q_recon_batch, pmu, plogvar = self.vaemodel.forward(data_z_aug)
        qmu = q_recon_batch[0]
        qlogvar = q_recon_batch[1]

        loss_rev_variational = self.loss_function_variationalmodel(qmu, qlogvar, data_x_rev_variational, data_z_aug,
                                                                   pmu, plogvar, N_z, N_xpz,
                                                                   x_dim=self.x_dim_mod, bgpu=self.gpu_mode,
                                                                   normalize=True, bpxOnly=False,
                                                                   beta_prefactor=beta_prefactor)

        self.vaemodel.train()
        return loss_rev_variational

    def assess_increase_kl_divergence(self, beta_pref_proposed, beta_pref_current):
        self.vaemodel.eval()
        with torch.no_grad():
            logdiff_z = self.md_importance_sampler.compute_log_diff_z_betaprime_m_zbeta(beta_pref_current, beta_pref_proposed)
            self.md_importance_sampler.store_current_log_z_diff(logdiff_z)

            # assess <\beta U(x)>_q(x,z)
            N_z = self.batch_size
            N_xpz = self.L
            data_z = torch.randn((N_z, self.z_dim), device=self.device)
            data_z_aug = data_z.repeat(N_xpz, 1)
            # ,mu and logvar of p(x|z)
            data_x_rev_variational, q_recon_batch, pmu, plogvar = self.vaemodel.forward(data_z_aug)
            m_beta_pref_curr_ux = self.MDLossapplied(data_x_rev_variational, self.refmodel, False, beta_pref_current, False)
            m_beta_pref_curr_ux = m_beta_pref_curr_ux.mean()

            beta_prop_m_beta_curr_times_ux = - (
                        beta_pref_proposed - beta_pref_current) * m_beta_pref_curr_ux / beta_pref_current

            loss_current_proportianal = self.assess_loss(beta_pref_current)
            log_z_current = self.md_importance_sampler.get_current_log_z(beta_pref_current)

            rel_kl_increas = (logdiff_z + beta_prop_m_beta_curr_times_ux) / (log_z_current + loss_current_proportianal)
        self.vaemodel.train()
        return rel_kl_increas

    def train_minibatch(self, bAVI=True):

        # just for Gaussian/Energy functional
        if self.name_model in ['gauss', 'quad']:
            lendata = 1
            lendataset = 1
            lendataloader = 1
        else:
            if type(self.data) is float:
                lendata = self.data
            else:
                lendata = len(self.data)
            lendataset = len(self.data_loader.dataset)
            lendataloader = len(self.data_loader)

        if (1. - self.weightvae) > 0.:
            N_z = self.batch_size
            N_xpz = self.L

            if self.gpu_mode:
                data_z = torch.randn((N_z, self.z_dim), device=torch.device('cuda'))
            else:
                data_z = torch.randn((N_z, self.z_dim))
            data_z_aug = data_z.repeat(N_xpz, 1)
            # TODO check this
            # data_z_aug.requires_grad = True

            # ,mu and logvar of p(x|z)
            data_x_rev_variational, q_recon_batch, pmu, plogvar = self.vaemodel.forward(data_z_aug)
            qmu = q_recon_batch[0]
            qlogvar = q_recon_batch[1]

            if self.bDebug:
                bNaN = np.zeros(6, dtype=bool)
                bNaN[0] = self.nanCheck(input=data_z, name='data_z')
                bNaN[1] = self.nanCheck(input=data_x_rev_variational, name='data_x')
                bNaN[2] = self.nanCheck(input=qmu, name='qmu')
                bNaN[3] = self.nanCheck(input=qlogvar, name='qlogvar')
                bNaN[4] = self.nanCheck(input=pmu, name='pmu')
                bNaN[5] = self.nanCheck(input=plogvar, name='plogvar')
                if np.any(bNaN):
                    print('NaN occurring.')
                    raise ValueError('NaN occurring: data_z data_x qmu qlogvar pmu plogvar {}'.format(bNaN))

                # print(np.exp(recon_logvar.data.cpu().numpy())
            loss_rev_variational = self.loss_function_variationalmodel(qmu, qlogvar, data_x_rev_variational, data_z_aug,
                                                                       pmu, plogvar, N_z, N_xpz,
                                                                       x_dim=self.x_dim_mod, bgpu=self.gpu_mode,
                                                                       normalize=True, bpxOnly=False)

            # output encoded x and decoded z
            if self.output_en_decoded and (not self.epoch % self.outputfrequ) and (self.batch_idx == 0) and bAVI:
                with torch.no_grad():
                    var = qlogvar.exp()
                    npvar = var.data.cpu().numpy()
                    name = 'encoded_x_var_' + str(self.epoch) + '.txt'
                    np.savetxt(os.path.join(self.output_dir, name), npvar)
                    name = 'encoded_x_mu_' + str(self.epoch) + '.txt'
                    np.savetxt(os.path.join(self.output_dir, name), qmu.data.cpu().numpy())
                    name = 'decoded_z_var_' + str(self.epoch) + '.txt'
                    var = plogvar.exp()
                    if self.breddescr:
                        var = convertToFullCartersianCoordinates(data=var)
                    np.savetxt(os.path.join(self.output_dir, name), var.data.cpu().numpy())
                    name = 'decoded_z_mu_' + str(self.epoch) + '.txt'
                    if self.breddescr:
                        pmu = convertToFullCartersianCoordinates(data=pmu)
                    np.savetxt(os.path.join(self.output_dir, name), pmu.data.cpu().numpy())
                    name = 'pred_x_' + str(self.epoch) + '.txt'
                    if self.breddescr:
                        data_x_rev_variational = convertToFullCartersianCoordinates(data=data_x_rev_variational)
                    np.savetxt(os.path.join(self.output_dir, name), data_x_rev_variational.data.cpu().numpy())

        if self.weightvae > 0.:

            # copy the data tensor for using more eps ~ p(eps) samples Eq. (7) in AEVB paper
            dataaug_vae = self.data.repeat(self.L, 1)
            if self.gpu_mode:
                dataaug_vae = dataaug_vae.cuda()

            if self.autoencvarbayes:
                recon_batch, mu, logvar = self.vaemodel.forward_vae(dataaug_vae)
                recon_mu = recon_batch[0]
                recon_logvar = recon_batch[1]

                # print(np.exp(recon_logvar.data.cpu().numpy())
                loss_vae = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, dataaug_vae, mu, logvar,
                                                              x_dim=self.x_dim_mod, normalize=True)
            else:
                recon_batch, mu, logvar = self.vaemodel.forward_vae(dataaug_vae)
                loss_vae = self.loss_function(recon_batch, dataaug_vae, mu, logvar, x_dim=self.x_dim_mod)

        if (1. - self.weightvae) > 0. and self.weightvae > 0.:
            loss_rev_variational = loss_rev_variational.mul(1. - self.weightvae)
            loss_rev_variational.backward(retain_graph=True)
            #
            # if self.bDebug:
            #    par = list(self.vaemodel.named_parameters())
            #    g = par[17][1].grad
            #    print('Grad from rev_var')
            #    print(g.norm())
            # clip the gradient
            nn.utils.clip_grad_value_(self.vaemodel.parameters(), 1.e11)
            # if self.bDebug:
            #    print(g)
            loss_vae = loss_vae.mul(self.weightvae)
            loss_vae.backward()
            # if self.bDebug:
            #    print('Grad rev_var + data')
            #    print(g.norm())
            loss = loss_rev_variational + loss_vae
            # loss = loss_rev_variational.mul(1.-weightvae) + loss_vae.mul(weightvae)
        elif self.weightvae == 1.:
            loss = loss_vae
            loss.backward()
            # if self.bDebug:
            #    par = list(self.vaemodel.named_parameters())
            #    g = par[17][1].grad
            #    print(g)
        else:
            loss = loss_rev_variational
            loss.backward()
            # if self.bDebug:
            #    par = list(self.vaemodel.named_parameters())
            #    g = par[17][1].grad
            #    print('Grad from rev_var')
            #    print(g.norm())

        self.vaemodel.write_gradient_norm()

        return loss, lendata, lendataset, lendataloader

    def trainepochCombined(self, epoch, weightvae):

        self.vaemodel.train()
        train_loss = 0

        # for batch_idx, (data, _) in enumerate(train_loader):
        # batch_idx could be replaced by iter (iteration during batch)

        if self.name_model in ['gauss', 'quad']:
            # Note: this only works for gaussian reference if no data is used.
            iterator = zip(np.arange(4), np.ones(4))
        else:
            iterator = zip(np.arange(2), np.ones(2))
            #iterator = self.data_loader

        # initialize variables
        self.epoch = 0
        self.weightvae = weightvae
        self.batch_idx = 0
        self.data = 1.

        if hasattr(self, 'laginstance'):
            self.laginstance.checkandoaggressiveupdate(epoch, False)

        for batch_idx, data in enumerate(iterator):
            self.epoch = epoch
            self.weightvae = weightvae
            self.batch_idx = batch_idx
            self.data = data

            # make sure gradient emptied from previous calculations
            self.optimizer.zero_grad()

            loss, lendata, lendataset, lendataloader = self.train_minibatch()

            # loss.backward(retain_graph=True)
            #loss.backward()
            train_loss += loss.item()

            #if (1. - weightvae) > 0:
            #    nn.utils.clip_grad_value_(self.vaemodel.parameters(), 1.e15)

            #if epoch > 2 and abs(loss.item()) > 5 * abs(self.train_hist['Total_loss'][-1]):
            #    continue
            #else:
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update step if we use reverse variational approach

            log_interval = 20
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * lendata, lendataset,
                           100. * batch_idx / lendataloader,
                           loss.item() ))

        self.train_hist['Total_loss'].append(loss.item())

        self.vaemodel.eval()


        # TODO rename variables self.name_model and self.model_name
        if (1. - weightvae) > 0.:
            self.refmodel.doStep()
            if hasattr(self.refmodel, 'schedBeta'):
                self.refmodel.schedBeta.updateLoss(self.train_hist['Total_loss'])
            if hasattr(self.refmodel, 'schedBetaCov'):
                self.refmodel.schedBetaCov.updateLoss(self.train_hist['Total_loss'])

        if hasattr(self.refmodel, 'schedBeta'):
            if hasattr(self, 'md_importance_sampler'):
                self.refmodel.schedBeta.updateLearningPrefactor(kl_current=loss.item(),
                                                                importance_sampler=self.md_importance_sampler,
                                                                gradient_norm=self.vaemodel.np_array_tot_grad)
            else:
                self.refmodel.schedBeta.updateLearningPrefactor(loss.item())
        if hasattr(self.refmodel, 'schedBetaCov'):
            self.refmodel.schedBetaCov.updateSigmaConv(loss.item())


    def trainepoch(self, epoch):

        self.vaemodel.train()
        train_loss = 0
        # for batch_idx, (data, _) in enumerate(train_loader):
        # batch_idx could be replaced by iter (iteration during batch)
        for batch_idx, data in enumerate(self.data_loader):

            # copy the data tensor for using more eps ~ p(eps) samples Eq. (7) in AEVB paper
            L = self.L
            dataaug = data.clone()
            for l in range(L - 1):
                dataaug = torch.cat((dataaug, data), 0)

            data = Variable(dataaug)
            if self.gpu_mode:
                data = data.cuda()

            self.optimizer.zero_grad()

            if self.autoencvarbayes:
                recon_batch, mu, logvar = self.vaemodel(data)
                recon_mu = recon_batch[0]
                recon_logvar = recon_batch[1]

                #print(recon_mu, recon_logvar, mu, logvar)
                #quit()

                # print(np.exp(recon_logvar.data.cpu().numpy()))
                loss = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, data, mu, logvar, x_dim=self.x_dim_mod)
            else:
                recon_batch, mu, logvar = self.vaemodel(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, x_dim=self.x_dim_mod)
            # loss.backward(retain_graph=True)
            loss.backward()
            train_loss += loss.item()

            self.optimizer.step()

            log_interval = 20
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.data_loader.dataset),
                           100. * batch_idx / len(self.data_loader),
                           loss.item() / len(data)))

        self.train_hist['Total_loss'].append(loss.item() / len(data))

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

                # print(np.exp(recon_logvar.data.cpu().numpy())
                loss = self.loss_function_autoencvarbayes(recon_mu, recon_logvar, data, mu, logvar, x_dim=self.x_dim_mod)
            else:
                recon_batch, mu, logvar = self.vaemodel(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, x_dim=self.x_dim_mod)

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
                        print(name)  # , param.data
                    else:
                        pert_param_list.append({'name': name, 'id': id, 'params': param})
                        pclone = param.clone()
                        params_dec_copy.append({'name': name, 'id': id, 'params': pclone})
                        print(name)  # , param.data
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

            #print(hess_params.size())
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
                hess_params[hess_params < 0.5] = 10.*self.N #1.e5
            hess_list.append({'name': parentry['name'], 'id': parentry['id'], 'diaghessian': hess_params.data, 'fullcov': bfullcov, 'parisvector': bparisvector})
            np.savetxt(os.path.join(self.output_dir, parentry['name']+'_' + str(self.N) + '.txt'), hess_params.data.cpu().numpy())

        self.vaemodel.bgetlogvar = bvartemp

        return pert_param_list, params_dec_copy, hess_list

    def getWeightVae(self, epoch):
        bSwitch = False
        bDirectSwitch = True
        epmin = 3000
        epsec = 105000

        if bSwitch:
            if epoch >= epmin:
                if bDirectSwitch:
                    weightvae = 0.0
                    # reinitialize the optimizer once we change the beta in the training
                    if epoch == epmin: # or epoch == epsec:
                        self.setOptimizer()
                        #self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)
                else:
                    if epoch == epmin:
                        self.weightvaeinit = 0.999
                        weightvae = 0.999
                        # reset the optimizer
                        self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)
                    elif epoch > epmin and epoch < epsec:
                        weightvae = (self.weightvaeinit - 0.5) / (epsec - epmin) * (epoch - epmin) + 0.5
                    elif epoch == epsec:
                        weightvae = 0.5
                        # reset the optimizer
                        self.setOptimizer()
                        #self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)
                    elif epoch > epsec:
                        weightvae = 0.5
            else:
                # set the weight of the vae versus the reverse variational approach
                weightvae = 1.0
        else:
            weightvae = 0.0

        return weightvae

    def train(self):

        self.train_hist = {}
        self.train_hist['Total_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['kl_qp'] = []

        if self.name_model in ['gauss', 'quad']:
            self.train_hist_pred = {}
            self.train_hist_pred['mu'] = []
            self.train_hist_pred['sig'] = []
            self.train_hist_pred['sig_error_norm'] = []
            self.train_hist_pred['mu_error_norm'] = []
            self.train_hist_pred['kl_ref_to_pred'] = []
            self.train_hist_pred['kl_pred_to_ref'] = []
            intermediate = True
            save_intermediate_model = True
        else:
            intermediate = False
            save_intermediate_model = True

        parlist = self.getweightlist()

        # if the model is loaded no need to train it.
        if not self.bloadmodel:
            print('training start!!')
            start_time = time.time()

            # set loss evaluation function in case of step scheduler depending on it.
            if hasattr(self.refmodel, 'schedBeta'):
                if self.abs_kl_increase_meassure:
                    self.refmodel.schedBeta.set_loss_evaluation_function(self.assess_increase_kl_divergence)
                else:
                    self.refmodel.schedBeta.set_loss_evaluation_function(self.assess_loss)

            if hasattr(self.refmodel, 'schedBetaCov'):
                self.refmodel.schedBetaCov.set_loss_evaluation_function(self.assess_loss)

            for epoch in range(1, self.epoch + 1):

                epoch_start_time = time.time()

                self.trainepochCombined(epoch, self.getWeightVae(epoch))
                # test(epoch)

                if hasattr(self.refmodel, 'schedBeta'):
                    if self.refmodel.schedBeta.resetoptimizer():
                        self.setOptimizer()
                    # self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3)

                if not epoch % self.outputfrequ or epoch == 1:
                    self.vaemodel.save_gradient_norm_list(self.output_dir)

                # check if \beta changed, then it is a new optimization problem and a the optimizer is to reset
                if self.name_model not in ['gauss', 'quad']:
                    # visualize intermediate latent space
                    self.visLatentTraining(epoch)

                    if not epoch % self.outputfrequ or epoch == 1:

                        samples = self.gen_samples(n_samples=self.n_samples, iter=epoch, bintermediate=True)

                        if hasattr(self.refmodel, 'propertycal'):
                            output_xtc = self.output_dir + '/samples_aevb' + self.predprefix + '_' + str(epoch) + '.xtc'
                            output_prefix = self.output_dir + '/samples_aevb' + self.predprefix + '_' + str(epoch)
                            self.refmodel.propertycal.write_trajectory(samples*self.unitlenghfactor, output_xtc)
                            try:
                                self.refmodel.propertycal.estimate_properties(output_xtc, output_prefix=output_prefix)
                            except:
                                print('Not able to estimate properties.')
                                np.savetxt(
                                    self.output_dir + '/samples_aevb' + self.predprefix + '_' + str(epoch) + '.txt',
                                    samples)
                                pass
                        # only store the plane txt file if not used directly for property calculation
                        else:
                            np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + '_' + str(epoch) + '.txt',
                                       samples)

                        np.savetxt(os.path.join(self.output_dir, 'train_hist.txt'),
                                   self.train_hist['Total_loss'])


                    # Make intermediate predictions during the training procedure
                    if intermediate:
                        # sample the prior
                        sample = Variable(torch.randn(64, self.z_dim, device=self.device))
                        # decode the samples
                        sample = self.vaemodel.decode(sample).cpu()
                        sampleout = convert_given_representation(samples=sample, coordrep=self.angulardata, bredcoord=self.breddescr)
                        np.savetxt(self.output_dir + '/samples' + self.predprefix + '.txt', sampleout)

                # Make intermediate predictions during the training procedure
                if intermediate and (not epoch % self.outputfrequ or epoch == 1):
                    self.intermediateVisGaussian(epoch=epoch, parlist=parlist)
                if save_intermediate_model and (not epoch % self.outputfrequ or epoch == 1):
                    #yborder = np.array([4., -4.])
                    # create predictions for the latent representation
                    #self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
                    # visualize the phi-psi landscape given the CVs
                    #self.vis_phipsilatent(path=self.output_dir)
                    torch.save(self.vaemodel, self.output_dir + '/model_' + str(epoch) + '.pth')

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

        if self.name_model in ['gauss','quad']:
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
        self.postprocessing()

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
        #x = np.linspace(-4., 4., 101)
        #y = np.linspace(-4., 4., 101)
        x = np.linspace(-2.5, 2.5, 101)
        y = np.linspace(-2.5, 2.5, 101)
        X, Y = np.meshgrid(x, y)
        Xvec = X.flatten()
        Yvec = Y.flatten()

        # convert numpy array to torch
        Xtorch = torch.from_numpy(Xvec).float()
        Xtorch.unsqueeze_(1)
        Ytorch = torch.from_numpy(Yvec).float()
        Ytorch.unsqueeze_(1)

        if self.gpu_mode:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1).cuda())
        else:
            samples_z = Variable(torch.cat((Xtorch, Ytorch), 1))

        # decode the CVs z to x
        torchsamples = self.vaemodel.decode(samples_z)
        # convert to numpy
        samples = torchsamples.data.cpu().numpy()

        # convert the samples if they are in the angular format
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, bredcoord=self.breddescr, unitgiven=self.unitlenghfactor)
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
        samples_aevb = np.zeros([n_samples_tot, self.x_dim_mod])

        icount = 0
        for n in range(n_samples):
            samples_aevb[n * nsamples_per_dp_tot, :] = mu[n, :]

            samples_aevb[n * nsamples_per_dp_tot + 1: (n + 1) * nsamples_per_dp_tot, :] = np.random.multivariate_normal(
                mu[n, :],
                np.diag(
                    var[n, :]/2.5),
                nsamples_per_dp)
        self.vaemodel.bgetlogvar = False

        samplesoutaevb = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata, bredcoord=self.breddescr)
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
        bVisualizeStar = True
        if bVisualizeStar == True:
            ny = 180
        bShowTraining = True
        if bShowTraining:
            xt = Variable(self.data_tensor)
        else:
            xt = None
        bRnd = False

        #y = torch.linspace(yb[0], yb[1], ny)

        # y coordinates
        y = torch.linspace(-1.5, 1.8, ny)
        #y1 = torch.linspace(-4, 0, ny*3)
        #y2 = torch.linspace(0, 4, ny*2)
        #nges =  5*ny
        #y = torch.cat((y1,y2))

        # x coordinates
        x = torch.zeros(ny) + 0.5

        # append samples for alpha mode
        nalpha = 61
        ny = ny + nalpha
        yalpha = torch.linspace(-1.5, 0.5, nalpha)
        xalpha = torch.zeros(nalpha) - 1.

        y = torch.cat((y, yalpha))
        x = torch.cat((x, xalpha))

        x_ref = Variable(self.data_tensor_vis_1527)
        if False:
            x = torch.linspace(-1, 3, ny)
            y12 = -2.*x
            y13 = -2./3.*x
            y = torch.zeros_like(y12)
            y.copy_(y12)
            y[20:] = y13[20:]

        # check if gpu mode is active
        if self.gpu_mode:
            y = y.cuda()
            x = x.cuda()
            x_ref = x_ref.cuda()

        # summarize x and y in torch variable
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)

        samples_z = Variable(torch.cat((y, x), 1))

        # This is for showing a little star at the current position in the latent space.
        # E.g. for visualizing atomistic configurations for given CVs.
        if bVisualizeStar:
            for i in range(0, ny):
                xnp_curr = samples_z[i, 0].data.cpu().numpy()
                ynp_curr = samples_z[i, 1].data.cpu().numpy()
                if i==0:
                    n = self.vaemodel.plotlatentrep(x=x_ref, z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=False, x_train=None) # x_train=xt
                else:
                    n = self.vaemodel.plotlatentrep(x=x_ref, z_dim=self.z_dim, path=self.output_dir,
                                            iter=i, x_curr=xnp_curr, y_curr=ynp_curr, nprov=True, normaltemp=n, x_train=None) # x_train=xt

        torchsamples = self.vaemodel.decode(samples_z)
        samples = torchsamples.data.cpu().numpy()

        # convert the samples if they are in the angular format
        samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, bredcoord=self.breddescr, unitgiven=self.unitlenghfactor)
        np.savetxt(path + '/samples_vis_latent' + self.predprefix + '.txt', samplesout)

    def gen_samples(self, n_samples=4000, postsampid=None, iter=-1, bintermediate=False):
        self.vaemodel.eval()
        # saving samples with postfix
        if postsampid is None:
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
                                                                   coordrep=self.angulardata, unitgiven=self.unitlenghfactor, bredcoord=self.breddescr)
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
            samplesout = convert_given_representation(samples=samples, coordrep=self.angulardata, unitgiven=self.unitlenghfactor, bredcoord=self.breddescr)
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
            samplesTorchmu, samplesTorchlogvar = self.vaemodel.decode(sample_z_)

            #print('mu')
            #print(samplesTorchmu)
            #print('samplesTorchlogvar')
            #print(samplesTorchlogvar)

            # TODO do not convert here to numpy for sampling from gaussian but use instead the torch implementation
            # of the Normal distribution. Those should be inculded in pyTorch 0.4.0
            mu = samplesTorchmu.data.cpu().numpy()
            vartorch = samplesTorchlogvar.exp()
            var = vartorch.data.cpu().numpy()

            #if self.bDebug:
            #np.savetxt(os.path.join(self.output_dir, 'var_pred.txt'), var)

            # init total amount of sample matrix
            n_samples_tot = n_samples * aevb_samples_per_mu
            samples_aevb = np.zeros([n_samples_tot, self.x_dim_mod])

            icount = 0
            # sample the p(x|z) for different CVs z and its corresponding \mu(z), \sigma(z).
            for n in range(n_samples):
                samples_aevb[n * aevb_samples_per_mu:(n + 1) * aevb_samples_per_mu, :] = np.random.multivariate_normal(
                    mu[n, :],
                    np.diag(
                        var[n, :]),
                    aevb_samples_per_mu)
            self.vaemodel.bgetlogvar = False

            # store the predictions
            if self.name_model not in ['gauss', 'quad']:
                samplesout = convert_given_representation(samples=samples_aevb, coordrep=self.angulardata,
                                                          unitgiven=self.unitlenghfactor, bredcoord=self.breddescr)
            else:
                samplesout = samples_aevb

            if not bintermediate:
                np.savetxt(self.output_dir + '/samples_aevb' + self.predprefix + postsamplepostfix + '.txt',
                       samplesout)

            #mean = samplesout.mean(axis=0)
            #std = samplesout.std(axis=0)
            #np.savetxt(self.output_dir + '/mean' + self.predprefix + postsamplepostfix + '_' + str(self.epoch) + '.txt', mean)
            #np.savetxt(self.output_dir + '/std' + self.predprefix + postsamplepostfix + '_' + str(self.epoch) + '.txt', std)

        return samplesout

    def visLatentTraining(self, epoch):
        if 'gauss' not in self.name_model:
            # Visualize intermediate steps, i.e. the latent embedding and the ELBO
            if self.bvislatent_training and not epoch % self.outputfrequ:
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

    def postprocessingALA15(self, postsampid=None):
        '''
        This function provides predictions given the trained model. In the case of \dim(z) = 2,
        further visualizations are issued automatically.
        :param n_samples: Amount of requred samples of z \sim p(z)
        :param postsampid: Do no specify this, it is just required internally for sampling the posterior of the decoding
        parametrization.
        :return:
        '''
        if hasattr(self.vaemodel, 'plotlatentrep') and postsampid is None:
            if self.gpu_mode:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                            path=self.output_dir, data_dir=self.data_dir, peptide=self.name_peptide)
            else:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir, x_train=Variable(self.data_tensor), data_dir=self.data_dir, peptide=self.name_peptide)

        # visualize latent representation if z_dim = 2
        if True and self.z_dim == 2 and postsampid is None:
            yborder = np.array([4., -4.])
            # create predictions for the latent representation
            self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
            # visualize the phi-psi landscape given the CVs
            #self.vis_phipsilatent(path=self.output_dir)

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
        if True and self.z_dim == 2 and postsampid is None and self.name_peptide == 'ala_2':
            #yborder = np.array([4., -4.])
            yborder = np.array([2., -1.5])
            # create predictions for the latent representation
            self.vis_latentpredictions(yb=yborder, ny=81, path=self.output_dir)
            # visualize the phi-psi landscape given the CVs
            self.vis_phipsilatent(path=self.output_dir)

        # visualize mapping between the different layers
        # TODO Add this for the vae model
        if hasattr(self.vaemodel, 'plotdecoder') and postsampid is None and self.name_peptide == 'ala_2':
            self.vaemodel.plotdecoder(n_samples=500, z_dim=self.z_dim)
        else:
            print('No visualization for decoder available.')

        # visualize the mapping from input to latent space
        if False and hasattr(self.vaemodel, 'plotencoder') and postsampid is None:
            data_loader_visualization = DataLoader(TensorDatasetDataOnly(self.data_tensor),
                                                   batch_size=1527,
                                                   shuffle=False, **self.kwargsdatloader)

            for index, data in enumerate(data_loader_visualization):
                data = Variable(data)
                if self.gpu_mode:
                    data = data.cuda()
                self.vaemodel.plotencoder(x=data, z_dim=self.z_dim, strindex=str(index))
        elif postsampid is None:
            print('No visualization for encoder available.')

        if hasattr(self.vaemodel, 'plotlatentrep') and postsampid is None:
            if self.gpu_mode:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527).cuda(), z_dim=self.z_dim,
                                            path=self.output_dir)
            else:
                self.vaemodel.plotlatentrep(x=Variable(self.data_tensor_vis_1527), z_dim=self.z_dim, path=self.output_dir, x_train=Variable(self.data_tensor))

            # store the variances of the test dataset
            varout = self.vaemodel.get_encoding_decoding_variance(x=Variable(self.data_tensor_vis_1527))
            temp_norm = np.append(varout['norm_enc'], varout['norm_dec'])
            np.savetxt(self.output_dir + '/normvar_enc_dec.txt', temp_norm)

        # visualize realizations along the z_1 or z_2 axis to show that the variance captures
        if False:
            self.vis_realizations()


    def nanCheck(self, input, name):
        nans = torch.isnan(input)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            print(name)
            print(input)
            return True
        else:
            return False

    def setOptimizer(self):
        '''Set the optimization scheme and distinguishes between different parameters if desired.'''
        if self.bseplearningrate:
            #listLogVar = ['dec_logvar', 'enc_logvar']
            listLogVar = ['dec_logvar']
            paramsExclLogVar = self.vaemodel.getExclParamList(listLogVar)
            paramsLogVar = self.vaemodel.getNamedParamList(listLogVar)

            self.optimizer = optim.Adam([{'params': paramsExclLogVar},
                                         {'params': paramsLogVar, 'lr': 1e-2}], lr=1e-3)
        else:
            self.optimizer = optim.Adam(self.vaemodel.parameters(), lr=1e-3, amsgrad=False)
            #from adam_clip import Adamclip
            #self.optimizer = Adamclip(self.vaemodel.parameters(), lr=1e-4, amsgrad=False, clamp_max_rel=0.04, clamp_eps=1e-6)
            ##self.optimizer = torch.optim.SGD(self.vaemodel.parameters(), lr=1e-3)

    def intermediateVisGaussian(self, epoch, parlist):

        if 'gauss' in self.name_model:

            np.savetxt(os.path.join(self.output_dir, 'train_hist.txt'), self.train_hist['Total_loss'])

            # save the parameters for this step
            #sstep = '%05d' % epoch
            #self.storeweightlist(parlist=parlist, path=self.output_dir, prefix=sstep)

            if hasattr(self.refmodel, 'plotencodedgaussian'):
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
                self.refmodel.plotencodedgaussian(vaemodel=self.vaemodel, samples=samp, p_val=p, z_rep=zmunp, path=self.output_dir,
                                                  postfix='_' + str(epoch), name_colormap='viridis',
                                                  mcomponents=self.refmodel.getMixtures())
                samples = self.gen_samples(n_samples=self.n_samples, iter=epoch, bintermediate=True)

                if self.x_dim == 2:
                    #mu, sig = self.vaemodel.plotprediction(samples=samples, path=self.output_dir, postfix='_' + str(epoch))  # , refsamples=self.refmodel.getRefSamples())
                    mu, sig = self.vaemodel.plotprediction(samples=samples, path=self.output_dir,
                                                                postfix='_' + str(
                                                                    epoch), refsamples=self.refmodel.getRefSamples())
                    if self.refmodel.getIsMixture():
                        mu_model = np.copy(mu)
                        cov_model_vector = np.copy(sig)
                    else:
                        if hasattr(self.vaemodel, 'getCov'):
                            cov_model = self.vaemodel.getCovNP()
                            elems = cov_model.size
                            cov_model_vector = np.reshape(cov_model, elems)
                        if hasattr(self.vaemodel, 'getMean'):
                            mu_model = self.vaemodel.getMeanNP()

                    self.train_hist_pred['mu'].append(mu_model)
                    self.train_hist_pred['sig'].append(cov_model_vector)
                    np.savetxt(os.path.join(self.output_dir, 'train_hist_mu.txt'), self.train_hist_pred['mu'])
                    np.savetxt(os.path.join(self.output_dir, 'train_hist_sig.txt'), self.train_hist_pred['sig'])

                # Analytic expressions available for single Gaussian.
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
                    covinvpred = np.linalg.inv(covpred)

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

                    # get KL-divergence KL( p_ref(x) || \bar p_\theta(x) )
                    # analytic solution available
                    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

                    mupredmmuref = mupred - muref

                    A = np.log(detcovpred / detcovref)
                    B = self.x_dim
                    C = np.trace(np.inner(covinvpred, covref))
                    D = np.inner(np.inner(mupredmmuref, covinvpred.T), mupredmmuref)
                    kldiv_q_to_ref = 0.5 * (A - B + C + D)
                    self.train_hist_pred['kl_pred_to_ref'].append(kldiv_q_to_ref)

                else:
                    #### KL PRED TO REF
                    # compute the kl divergence between distributions
                    # we need to approximate it
                    logp = self.refmodel.getlogpdfTorch(samples_torch)
                    bgettemp = self.vaemodel.bgetlogvar
                    self.vaemodel.bgetlogvar = True
                    logbarptheta = self.vaemodel.getlogpdf(samples=samples_torch, bgpu=self.gpu_mode, nz=300)
                    self.vaemodel.bgetlogvar = bgettemp
                    if self.gpu_mode:
                        KLdist = - torch.mean(logbarptheta.cuda() - logp.cuda())
                    else:
                        KLdist = - torch.mean(logbarptheta - logp)
                    KLdist = KLdist.data.cpu().numpy()
                    #if type(KLdist) is np.ndarray:
                    #    KLdist = KLdist.astype(np.float64)
                    #    KLdist = KLdist.item()
                    self.train_hist_pred['kl_pred_to_ref'].append(KLdist)

                    #### KL REF TO PRED
                    # samples from pred are required
                    if self.gpu_mode:
                        samples_pred_troch = torch.from_numpy(samples).cuda()
                    else:
                        samples_pred_troch = torch.from_numpy(samples)
                    samples_pred_troch = samples_pred_troch.type(torch.float)
                    logp = self.refmodel.getlogpdfTorch(samples_pred_troch)
                    bgettemp = self.vaemodel.bgetlogvar
                    self.vaemodel.bgetlogvar = True
                    logbarptheta = self.vaemodel.getlogpdf(samples=samples_pred_troch, bgpu=self.gpu_mode, nz=300)
                    self.vaemodel.bgetlogvar = bgettemp
                    if self.gpu_mode:
                        KLdist = - torch.mean(logp.cuda() - logbarptheta.cuda())
                    else:
                        KLdist = - torch.mean(logp - logbarptheta)
                    KLdist = KLdist.data.cpu().numpy()
                    #if type(KLdist) is np.ndarray:
                    #    KLdist = KLdist.astype(np.float64)
                    #    KLdist = KLdist.item()
                    self.train_hist_pred['kl_ref_to_pred'].append(KLdist)


                x = range(0, epoch + 1, self.outputfrequ)
                utils.plt_kldiv(x=x, y=self.train_hist_pred['kl_pred_to_ref'], path=self.output_dir,
                                filename='KL_pred_to_ref.pdf', labelname=r'$D_{\text{KL}}( p_{traget}(\mathrm{x}) || q_{\theta}(\mathrm{x}) )$')
                utils.plt_kldiv(x=x, y=self.train_hist_pred['kl_ref_to_pred'], path=self.output_dir,
                                filename='KL_ref_to_pred.pdf', labelname=r'$D_{\text{KL}}( q_{\theta}(\mathrm{x}) || p_{traget}(\mathrm{x}) )$')
                utils.plt_kldiv_combined(x=x, y=[self.train_hist_pred['kl_pred_to_ref'], self.train_hist_pred['kl_ref_to_pred']], path=self.output_dir,
                                filename='KL_combined.pdf', labelname=[r'$D_{\text{KL}}( p_{traget}(\mathrm{x}) || q_{\theta}(\mathrm{x}) )$', r'$D_{\text{KL}}( q_{\theta}(\mathrm{x}) || p_{traget}(\mathrm{x}) )$'])

                np.savetxt(os.path.join(self.output_dir, 'train_hist_kl_pred_to_ref.txt'),
                           self.train_hist_pred['kl_pred_to_ref'])

            np.savetxt(os.path.join(self.output_dir, 'train_hist_sig_error.txt'), self.train_hist_pred['sig_error_norm'])
            np.savetxt(os.path.join(self.output_dir, 'train_hist_mu_error.txt'), self.train_hist_pred['mu_error_norm'])
            np.savetxt(os.path.join(self.output_dir, 'train_hist_kl.txt'), self.train_hist_pred['kl_ref_to_pred'])
            ##############################################################

        elif 'quad' in self.name_model:

            np.savetxt(os.path.join(self.output_dir, 'train_hist.txt'),
                       self.train_hist['Total_loss'])

            # save the parameters for this step
            #sstep = '%05d' % epoch
            #self.storeweightlist(parlist=parlist, path=self.output_dir, prefix=sstep)

            # produce samples from reference model
            # samples_torch = self.refmodel.sample_torch(nsamples=2000)
            samples_torch = self.refmodel.getRefSamples(10)

            if hasattr(self.refmodel, 'plotencodedrepresentaion'):
                #if self.gpu_mode:
                #    samples_torch = samples_torch.cuda()

                zmu, zlogvar = self.vaemodel.encode(samples_torch)

                # plot the encoded samples
                sigma = torch.sqrt(zlogvar.exp())
                z_sample = zmu.clone()
                nreplicates = 5
                for i in range(nreplicates):
                    z_sample = torch.cat((z_sample, torch.randn_like(zmu) * sigma + zmu), 0)
                xs = samples_torch.repeat(nreplicates+1, 1)

                if self.gpu_mode:
                    p = self.refmodel.getpdf(xs).cpu().data.numpy()
                    zmunp = z_sample.cpu().data.numpy()
                    samp = xs.cpu().data.numpy()
                else:
                    p = self.refmodel.getpdf(xs).data.numpy()
                    zmunp = z_sample.data.numpy()
                    samp = xs.data.numpy()

                self.refmodel.plotencodedrepresentaion(samples=samp, p_val=p, z_rep=zmunp, path=self.output_dir, postfix='_' + str(epoch))
                samples = self.gen_samples(n_samples=self.n_samples, iter=epoch, bintermediate=True)

                mean_intermed = samples.mean(axis=0)
                std_intermed = samples.std(axis=0)

                if not (hasattr(self, 'list_means') or hasattr(self, 'list_std')):
                    self.list_means = []
                    self.list_std = []
                self.list_means.append(mean_intermed)
                self.list_std.append(std_intermed)

                np.savetxt(
                    self.output_dir + '/mean' + '.txt',
                    self.list_means)
                np.savetxt(
                    self.output_dir + '/std' + '.txt',
                    self.list_std)

                self.refmodel.vis_latentpredictions(vaemodel=self.vaemodel, path=self.output_dir,
                                                        postfix='_' + str(epoch), do_latent_movement=False)#((epoch % 50) == 0))

            if hasattr(self.vaemodel, 'plotprediction'):
                if hasattr(self.refmodel, 'getRefWeights'):
                    weights = self.refmodel.getRefWeights()
                    if weights is not None:
                        weights = weights.cpu().data.numpy().T

                    self.vaemodel.plotprediction(samples=samples, path=self.output_dir, postfix='_' + str(epoch),
                                                 refsamples=self.refmodel.getRefSamples().cpu().data.numpy().T, refsamples_weights=weights)
                else:
                    self.vaemodel.plotprediction(samples=samples, path=self.output_dir, postfix='_' + str(epoch),
                                                       refsamples=self.refmodel.getRefSamples().cpu().data.numpy().T)

            # actually this is not really relevant for energy functional
            self.train_hist_pred['mu'].append(0.)
            self.train_hist_pred['sig'].append(0.)
            np.savetxt(os.path.join(self.output_dir, 'train_hist_mu.txt'), self.train_hist_pred['mu'])
            np.savetxt(os.path.join(self.output_dir, 'train_hist_sig.txt'), self.train_hist_pred['sig'])

            nsize = samples.shape[0]
            if nsize > 25000:
                nskip = int(nsize/25000)
                samples = samples[0:-1:nskip, :]

            #### KL PRED TO REF
            # compute the kl divergence between distributions
            # we need to approximate it
            #logp = self.refmodel.logpx(samples_torch)
            logp = self.refmodel.getlogpdfTorch(samples_torch)
            logbarptheta = self.vaemodel.getlogpdf(samples=samples_torch, bgpu=self.gpu_mode, nz=300)
            if self.gpu_mode:
                KLdist = - torch.mean(logbarptheta.cuda() - logp.cuda())
            else:
                KLdist = - torch.mean(logbarptheta - logp)
            KLdist = KLdist.data.cpu().numpy()
            #if type(KLdist) is np.ndarray:
            #    KLdist = KLdist.astype(np.float64)
            #    KLdist = KLdist.item()
            self.train_hist_pred['kl_pred_to_ref'].append(KLdist)

            #### KL REF TO PRED
            # samples from pred are required
            samples_pred_troch = torch.from_numpy(samples).float().to(self.device)
            # assess at current beta sched if scheduler is available
            logp = self.refmodel.getlogpdfTorch(samples_pred_troch)
            logbarptheta = self.vaemodel.getlogpdf(samples=samples_pred_troch, bgpu=self.gpu_mode, nz=300)
            if self.gpu_mode:
                KLdist = - torch.mean(logp.cuda() - logbarptheta.cuda())
            else:
                KLdist = - torch.mean(logp - logbarptheta)
            KLdist = KLdist.data.cpu().numpy()
            #if type(KLdist) is np.ndarray:
            #    KLdist = KLdist.astype(np.float64)
            #    KLdist = KLdist.item()
            self.train_hist_pred['kl_ref_to_pred'].append(KLdist)

            if hasattr(self.refmodel, 'schedBeta'):
                beta = self.refmodel.schedBeta.getLearningPrefactor()
            else:
                beta = 1.0

            self.refmodel.plotpotentials(vaemodel=self.vaemodel, path=self.output_dir, postfix='_' + str(epoch), beta=beta)

            if len(self.train_hist_pred['kl_pred_to_ref'][1:]) > 2:
                x = range(0, epoch, self.outputfrequ)
                utils.plt_kldiv(x=x, y=self.train_hist_pred['kl_pred_to_ref'][1:], path=self.output_dir,
                                filename='KL_pred_to_ref.pdf', labelname=r'$D_{\text{KL}}( p_\text{target}(\mathbf{x}) || q_{\boldsymbol{\theta}}(\mathbf{x}) )$')
                utils.plt_kldiv(x=x, y=self.train_hist_pred['kl_ref_to_pred'][1:], path=self.output_dir,
                                filename='KL_ref_to_pred.pdf', labelname=r'$D_{\text{KL}}( q_{\boldsymbol{\theta}}(\mathbf{x}) || p_\text{target}(\mathbf{x})  )$')
                utils.plt_kldiv_combined(x=x, y=[self.train_hist_pred['kl_pred_to_ref'][1:], self.train_hist_pred['kl_ref_to_pred'][1:]], path=self.output_dir,
                                filename='KL_combined.pdf', labelname=[r'$D_{\text{KL}}( p_\text{target}(\mathbf{x}) || q_{\boldsymbol{\theta}}(\mathbf{x}) )$', r'$D_{\text{KL}}( q_{\boldsymbol{\theta}}(\mathbf{x}) || p_\text{target}(\mathbf{x})  )$'])

                np.savetxt(os.path.join(self.output_dir, 'train_hist_kl_pred_to_ref.txt'),
                           self.train_hist_pred['kl_pred_to_ref'])

                np.savetxt(os.path.join(self.output_dir, 'train_hist_sig_error.txt'), self.train_hist_pred['sig_error_norm'])
                np.savetxt(os.path.join(self.output_dir, 'train_hist_mu_error.txt'), self.train_hist_pred['mu_error_norm'])
                np.savetxt(os.path.join(self.output_dir, 'train_hist_kl_pred_to_ref.txt'), self.train_hist_pred['kl_pred_to_ref'])
                np.savetxt(os.path.join(self.output_dir, 'train_hist_kl_ref_to_pred.txt'), self.train_hist_pred['kl_ref_to_pred'])
            ##############################################################