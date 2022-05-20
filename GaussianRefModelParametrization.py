
import torch


class GaussianRefModelParametrization:
    '''
    This class provides a parametrization for a target Guassian model (either mixture or single Gaussian).
    '''

    @staticmethod
    def getParVectors(x_dim, z_dim, nModes, bassigrandW):
        with torch.no_grad():
            if x_dim == 2 and nModes == 1:
                # assign two dimensional guassian as target model

                # muref = torch.tensor([0.25, 0.7])
                muref = torch.tensor([0.25, 0.7])
                # sigmaref = torch.tensor([0.05, 0.01])
                #sigmaref = torch.tensor([0.1, 0.5])
                #W_ref = torch.tensor([10., -0.2])

                sigmaref = torch.tensor([0.01, 0.05])
                W_ref = torch.tensor([.1, -0.03])

                # sigmaref = torch.tensor([0.1, 0.5])
                # W_ref =  torch.tensor([10., -0.2])
            elif x_dim > 2 and nModes == 1:
                # Assign multidimensional gaussian as target model

                bWassignRadnom = bassigrandW

                # reference mu
                muref = torch.rand(x_dim).add(-0.5).mul(4)

                # sigma sq
                sig_values = torch.tensor([0.8, 0.5, 0.4, 0.3]).mul(2.)
                indSig = torch.LongTensor(x_dim).random_(0, 4)
                sigmaref = torch.zeros(x_dim)
                for idx, sig in enumerate(sigmaref):
                    sigmaref[idx] = sig_values[indSig[idx]]

                if not bWassignRadnom:
                    W_values = torch.tensor([1, 0.5, 1.5, -1, -0.5, 1.5, -1.2])

                    indW = torch.LongTensor(x_dim, z_dim).random_(0, 7)
                    W_ref = torch.zeros(x_dim, z_dim)
                    for idx, w in enumerate(W_ref):
                        W_ref[idx] = W_values[indW[idx]]
                else:
                    W_ref = torch.rand(x_dim).add(-0.5).mul(0.5)
                # sigmaref = torch.rand(100).add(-0.5).mul(1)
                #
            elif x_dim == 2 and nModes == 2:
                # Specify mixture model if x_dim == 2

                # muref = torch.tensor([0.25, 0.7])
                muref = torch.tensor([[-0.5, -0.5], [0.25, 0.7]])
                # sigmaref = torch.tensor([0.05, 0.01])
                sigmaref = torch.tensor([[0.1, 0.2], [0.2, 0.1]])
                W_ref = torch.tensor([[0.1, -0.2], [0.3, 0.1]])

                # muref = torch.tensor([[-0.5, -0.5], [0.5, 0.6]])
                ## sigmaref = torch.tensor([0.05, 0.01])
                sigmaref = torch.tensor([[0.01, 0.02], [0.05, 0.01]])
                W_ref = torch.tensor([[0.05, -0.2], [0.2, 0.1]])
                # sigmaref = torch.tensor([[0.1, 0.2], [0.1, 0.1]])
                # W_ref =  torch.tensor([[0.01, -0.1], [0.1, 0.04]])
                # sigmaref = torch.tensor([0.1, 0.5])
                # W_ref =  torch.tensor([10., -0.2])
            elif x_dim > 2 and nModes == 2:
                # here we assume z_dim = 1, one might need to extend this however.

                # reference mu
                muref = torch.rand(nModes, x_dim).add(-0.5).mul(4)

                # sigma sq
                sig_values = torch.tensor([0.8, 0.5, 0.2, 0.1]).mul(1.)
                indSig = torch.LongTensor(nModes, x_dim).random_(0, 4)
                sigmaref = torch.zeros(nModes, x_dim)
                # for idx, sig in enumerate(sigmaref):
                for idx in range(x_dim):
                    for nmod in range(nModes):
                        sigmaref[nmod, idx] = sig_values[indSig[nmod, idx]]

                if not bassigrandW:
                    W_values = torch.tensor([0.9, 0.01, 1.5, -1, -0.4, 1.2, -0.2])

                    indW = torch.LongTensor(nModes, x_dim).random_(0, 7)
                    W_ref = torch.zeros(nModes, x_dim)
                    # for idx, w in enumerate(W_ref):
                    for idx in range(x_dim):
                        for nmod in range(nModes):
                            W_ref[nmod, idx] = W_values[indW[nmod, idx]]
                else:
                    W_ref = torch.rand(nModes, x_dim).add(-0.5).mul(0.5)
                # sigmaref = torch.rand(100).add(-0.5).mul(1)
                #
        return muref, sigmaref, W_ref