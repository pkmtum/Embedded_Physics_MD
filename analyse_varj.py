
import argparse, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():

    desc = "Analyse results of variational appraoch."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--ref_mean', type=str,
                        help='The mean of the reference gaussian', required=False)
    parser.add_argument('--ref_std', type=str,
                        help='The standard deviation of the reference gaussian', required=False)
    parser.add_argument('--outputfreq', type=int, required=True,
                        help='This specifies the output frequency for plotting the proper iterations on the x-axis.')

    return parser.parse_args()

def plot_error(iter, kl, ermu, ersig):

    f, ax = plt.subplots(2, 1, sharex=True)

    ax[0].semilogy(iter, kl)
    ax[0].grid(ls='dashed')
    ax[0].set_ylabel(r'$KL(\bar p_{\theta}(x) || p(x) )$')
    ax[0].axhline(y=0,xmin=0,xmax=iter[-1])

    ax[1].semilogy(iter, ermu, label=r'$\Delta_{\mu}$')
    ax[1].semilogy(iter, ersig, label=r'$\Delta_{\Sigma}$')
    ax[1].grid(ls='dashed')
    ax[1].set_ylabel('Relative Error Per Dimension')
    ax[1].axhline(y=0, xmin=0, xmax=iter[-1])
    ax[1].legend()
    ax[1].set_xlabel('Iteration')


    f.savefig('plt_error.pdf', bbox_layout='tight')


def plot_iterationsvsref(mean, std, refmean, refstd, elbo):

    f, ax = plt.subplots(3, 1, sharex=True)

    x = elbo[:, 0]

    ax[0].plot(x, mean[:, 0], label=r'$\mu_1$')
    ax[0].plot(x, mean[:, 1], label=r'$\mu_2$')
    ax[0].axhline(y=refmean[0], ls='-.', label=r'$\mu_{ref,1}$', alpha=0.5)
    ax[0].axhline(y=refmean[1], ls='-.', label=r'$\mu_{ref,2}$', alpha=0.5)
    ax[0].set_ylabel(r'$\mu$')
    ax[0].legend()
    ax[1].semilogy(x, std[:, 0], label=r'$\sigma_1$')
    ax[1].semilogy(x, std[:, 1], label=r'$\sigma_2$')
    ax[1].axhline(y=refstd[0], ls='-.', label=r'$\sigma_{ref,1}$', alpha=0.5)
    ax[1].axhline(y=refstd[1], ls='-.', label=r'$\sigma_{ref,2}$', alpha=0.5)
    ax[1].set_ylabel(r'$\sigma$')
    ax[1].legend()

    ax[2].plot(x, elbo[:, 1])
    ax[2].set_ylabel('ELBO')

    f.savefig('plt_convergence.pdf')
    plt.close(f)

def main():

    args = parse_args()

    #refmean = np.fromstring(args.ref_mean, sep=' ')
    #refstd = np.fromstring(args.ref_std, sep=' ')

    #itermean = np.loadtxt('train_hist_mu.txt')
    #iterstd = np.loadtxt('train_hist_sig.txt')
    #elbo = np.loadtxt('VARjoint_gauss_loss.txt')[0::args.outputfreq]

    kl = np.loadtxt('train_hist_kl.txt')
    error_mu = np.loadtxt('train_hist_mu_error.txt')[:, 1]
    error_sig = np.loadtxt('train_hist_sig_error.txt')[:, 1]

    itertot = (kl.shape[0] - 1) * args.outputfreq
    iter = np.arange(0, itertot+1, args.outputfreq)
    iter[0] = 1
    print iter.shape
    print kl.shape
    plot_error(iter, kl, error_mu, error_sig)



    #plot_iterationsvsref(mean=itermean, std=iterstd, refmean=refmean, refstd=refstd, elbo=elbo)

    print 'This is the main function.'
    quit()


if __name__ == '__main__':
    main()