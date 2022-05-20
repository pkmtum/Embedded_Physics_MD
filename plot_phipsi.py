import argparse, os
import numpy as np
import matplotlib.pyplot as plt



def plotphipsi(inputXVG):
    incr = 21
    skiprama = 32
    phipsi = np.loadtxt(inputXVG, delimiter='  ', skiprows=skiprama, usecols=range(0, 2))

    phi = phipsi[:, 0]
    psi = phipsi[:, 0]

    phi = phi.reshape(incr,incr)
    psi = psi.reshape(incr, incr)

    x = np.linspace(-4., 4., incr)
    y = np.linspace(-4., 4., incr)
    X, Y = np.meshgrid(x, y)

    plt.figure(1)
    f, ax = plt.subplots()
    #plt.contour(X, Y, Z, lw=.5, alpha=0.5)
    ax.contour(X, Y, phi)
    f.savefig('phi.pdf', bbox_inches='tight')
    f.close()

    plt.figure(1)
    f, ax = plt.subplots()
    #plt.contour(X, Y, Z, lw=.5, alpha=0.5)
    ax.contour(X, Y, psi)
    f.savefig('psi.pdf', bbox_inches='tight')
    f.close()


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation for creating plots of latent space"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--ramaxvg', type=str, default='',
                        help='File where phi psi angles are stored.', required=True)
    return parser.parse_args()

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    plotphipsi(inputXVG=args.ramaxvg)