import sys
import argparse, os

from WGAN_peptide import WGANPeptide
from GAN_peptide import GANPeptide
#from VAE import VAEpeptide
from VAE_up import VAEpeptide


"""get choices for implemented dataset"""
def get_choices():

    choice_list = ['mnist', 'fashion-mnist', 'celebA', 'samples', 'm_526', 'm_10437',
                   'a_500', 'b1_500', 'b2_500', 'a_1000', 'b1_1000', 'b2_1000',
                   'a_10000', 'b1_10000', 'b2_10000', 'var_gauss', 'ala_2', 'quad']
    choice_list.append('m_52')
    choice_list.append('m_102')
    choice_list.append('m_526')
    choice_list.append('m_262')
    choice_list.append('m_1001')
    choice_list.append('ma_10')
    choice_list.append('ma_50')
    choice_list.append('ma_100')
    choice_list.append('ma_200')
    choice_list.append('ma_500')
    choice_list.append('ma_1500')
    choice_list.append('ma_4000')
    choice_list.append('ma_13334')
    choice_list.append('m_ala_15')
    choice_list.append('m_100_ala_15')
    choice_list.append('m_200_ala_15')
    choice_list.append('m_300_ala_15')
    choice_list.append('m_500_ala_15')
    choice_list.append('m_1500_ala_15')
    choice_list.append('m_3000_ala_15')
    choice_list.append('m_5000_ala_15')
    choice_list.append('m_10000_ala_15')
    choice_list.append('m_20000_ala_15')


    for strN in ['1527', '4004']:
        choice_list.append('m_'+strN)
        choice_list.append('b1b2_' + strN)
        choice_list.append('ab1_' + strN)
        choice_list.append('ab2_' + strN)
    return choice_list

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_type', type=str, default='VARjoint',
                        choices=['GAN_peptide', 'VAE', 'VARjoint'],
                        help='The type of GAN')#, required=True)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=get_choices(),
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0001)
    parser.add_argument('--lrD', type=float, default=0.0001)
    parser.add_argument('--lrAll', type=float, default=0.0001)
    parser.add_argument('--lrLogVar', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=int, default=0)
    parser.add_argument('--clusterND', type=int, default=0)
    parser.add_argument('--cluster', type=str, choices=[None, 'ND', 'TUM'], default=None)
    parser.add_argument('--outPostFix', type=str, default='')
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--clipping', type=float, default=0.01)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--samples_pred', type=int, default=4000)
    parser.add_argument('--useangulardat', type=str, default='no',
                        choices=['no', 'ang', 'ang_augmented', 'ang_auggrouped'])
    parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0), 0 for no seed.')

    parser.add_argument('--AEVB', type=int, default=0,
                    help='Use Auto-Encoding Variational Bayes? If not, formulation relates to adversarial learning.')
    parser.add_argument('--Z', type=int, default=1,
                        help='Relevant for variational approach. It specifies the amount of samples from p(z).')
    parser.add_argument('--L', type=int, default=1,
                    help='Samples from eps ~ p(eps) for VAE.')
    parser.add_argument('--samples_per_mean', type=int, default=0,
                    help='Amount of predictive samples for p(x|z) = N(mu(z), sigma(z)). If 0, mean prediction is used: mu(z).')
    parser.add_argument('--npostS', type=int, default=0, help='Amount of posterior samples.')
    parser.add_argument('--uqbias', type=int, default=0, help='Quantify uncertainty of bias terms in network.')
    parser.add_argument('--exppriorvar', type=float, default=35., help='lambda of exp(-lambda theta. If 0, no prior employed')
    parser.add_argument('--sharedlogvar', type=int, default=1,
                        help='Sharing the logvariance instead of cosidering a variance dpendent on the decoding network.')
    parser.add_argument('--sharedencoderlogvar', type=int, default=0,
                        help='Sharing the logvariance of the ENCODER, instead of cosidering a variance dpendent on the encoding network. This only applies for VARJ.')
    parser.add_argument('--loadtrainedmodel', type=str, default='',
                        help='Provide the path including file of an already trained model for doing predictions.')
    parser.add_argument('--loadstatedict', type=str, default='',
                        help='Provide the path including model and only load the state dict from it.')
    parser.add_argument('--ard', type=float, default=0., help='Value of a0 for ARD prior. If 0. then no ARD prior is applyed.')
    parser.add_argument('--exactlikeli', type=int, default=0, help='Perform leveraging the exact likelihood.')
    parser.add_argument('--outputfreq', type=int, default=500, help='Output frequency during the optimization process.')
    parser.add_argument('--x_dim', type=int, default=2, help='Just for variational approach. Test to predict gaussian of dim x_dim.')
    parser.add_argument('--assignrandW', type=int, default=0,
                        help='Just for variational approach. Assign uniformly random variables to reference W.')
    parser.add_argument('--freeMemory', type=int, default=0,
                        help='Just for variational approach. Free memory during estimation of the loss function.')
    parser.add_argument('--setBetaPrefactor', type=float, default=None, help='Set the initial beta prefactor a*\\beta.')
    parser.add_argument('--stepSched', type=int, default=1, help='Use step scheduler module druing optimization. Set integer value for Max steps.')
    parser.add_argument('--stepSchedresetopt', type=int, default=0, help='Reset the optimizer in case an annealing step has been performed.')
    parser.add_argument('--stepSchedintwidth', type=float, default=0.005, help='Convergence criterion for making a step in the scheduler.')
    parser.add_argument('--stepSchedType', type=str, default='kl', choices=['kl', 'lin'], help='Select scheduler type.')
    parser.add_argument('--betaVAE', type=float, default=1., help='Beta value for enforcing beta * KL(q(z|x) || p(z)). See https://openreview.net/pdf?id=Sy2fzU9gl.')
    parser.add_argument('--separateLearningRate', type=int, default=0, help='This applies to separate learning rates between NN parameters and the parameters for the variances. Applies only if en- or decoding variance is modeled as parameter.')
    parser.add_argument('--redDescription', type=int, default=0, help='Only relevant for reverse var. approach. This removes 6 DOFs from x to implicitly remove the rigid body motion.')
    parser.add_argument('--laggingInferenceStepsPhi', type=int, default=0, help='This enables avoiding lagging inference. Number of updates before joint updates for phi.')
    parser.add_argument('--laggingInferenceStepsTheta', type=int, default=0, help='This enables avoiding lagging inference. Number of updates before joint updates for phi.')
    parser.add_argument('--output_path', type=str, default='', help='Defines the output path.')
    parser.add_argument('--md_grad_postproc', type=str, default='none', choices=['none', 'gradclamp', 'gradnorm', 'gradnormsingle', 'gradnormmax', 'monitor'], help='Specifies how to postprocess the gradient.')
    parser.add_argument('--add_reference_gaussian', type=int, default=0, choices=[0, 1], help='Adds a Gaussian around a reference configuration.')
    parser.add_argument('--add_gaussian_sig_sq', type=float, default=0., help='Adds a Gaussian with zero mean.')
    parser.add_argument('--convolute_target_potential_sig', type=float, default=0., help='Convolution of potential with Gaussian for smoothing. Provide sigma of gaussian.')
    parser.add_argument('--convolute_target_potential_n', type=int, default=0., help='Convolution of potential with Gaussian for smoothing. Provide amount of samples.')
    parser.add_argument('--convergence_criterion', type=str, default='check_loss_increase',
                        choices=['check_loss_increase', 'check_loss_band', 'check_grad_norm'])
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Select dropout rate, if 0 (default) dropout is off.')
    parser.add_argument('--dropout_enc_dec', type=str, default='dec',
                        choices=['enc_dec', 'dec_enc', 'dec', 'enc'],
                        help='Choose where dropout should be applied. This has only an effect if dropout > 0.')
    parser.add_argument('--max_kl_inc', default=1.e-5,
                        help='Default maximally allowed relative increase of the KL-Divergence.')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    try:
        if bool(args.add_reference_gaussian) and not bool(args.add_gaussian_sig_sq):
            assert False
        else:
            assert True
    except:
        print('If a reference structure is added, specify the variance of the Gaussian with --add_gaussian_sig_sq.')

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --laggingInferenceStepsPhi and --laggingInferenceStepsTheta
    if args.laggingInferenceStepsPhi < 0 or args.laggingInferenceStepsTheta < 0:
        raise Warning('Lagging inference steps is inactive due to wrong step arguments.')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # declare instance, though this code only should be used for VARjoint
    if args.model_type == 'GAN_peptide':
        model = GANPeptide(args)
    elif args.model_type == 'VAE':
        model = VAEpeptide(args)
    elif args.model_type == 'VARjoint':
        from VARJ import VARjoint
        model = VARjoint(args, sys.argv)
    else:
        raise Exception("[!] There is no option for " + args.model_type)

        # launch the graph in a session
    model.train()
    print(" [*] Training finished!")

    # visualize learned generator
    # model.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
