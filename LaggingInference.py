import torch
import os
import numpy as np

class LaggingInference:
    def __init__(self, modelobject, steps_theta=10, steps_phi=20):
        self.aggressiveupdate = True

        self.modelobject = modelobject

        self.params_enc = modelobject.vaemodel.getNamedParamList('enc_')
        self.params_dec = modelobject.vaemodel.getNamedParamList('dec_')

        self.optimizer_enc = torch.optim.Adam(self.params_enc, lr=1e-3)
        self.optimizer_dec = torch.optim.Adam(self.params_dec, lr=1e-3)

        self.stepsupdatephi = steps_phi
        self.stepsupdatetheta = steps_theta

    def checkandoaggressiveupdate(self, epoch, verbose):
        if self.aggressiveupdate and epoch > 2:
            self.modelobject.vaemodel.setRequiresGrad(parameter_list=self.params_dec, requires_grad=False)
            self.performoptimization(self.optimizer_enc, self.stepsupdatephi, variables='phi', epoch=epoch,
                                     verbose=verbose)
            self.modelobject.vaemodel.setRequiresGrad(parameter_list=self.params_dec, requires_grad=True)

            self.modelobject.vaemodel.setRequiresGrad(parameter_list=self.params_enc, requires_grad=False)
            self.performoptimization(self.optimizer_dec, self.stepsupdatetheta, variables='theta', epoch=epoch,
                                     verbose=verbose)
            self.modelobject.vaemodel.setRequiresGrad(parameter_list=self.params_enc, requires_grad=True)

    def performoptimization(self, opt, steps, variables, epoch, verbose=False):
        loss_list = []
        for i in range(steps):
            opt.zero_grad()
            loss, lendata, lendataset, lendataloader = self.modelobject.train_minibatch(bAVI=False)
            loss_list.append(loss.item())
            opt.step()
        if verbose:
            self.writeloss(loss_list, 'lagupdate_' + variables + '_', epoch)


    def writeloss(self, loss_list, prefix, epoch):
        filename = prefix + str(epoch) + '.txt'
        pathfile = os.path.join(self.modelobject.output_dir, filename)
        np.savetxt(pathfile, loss_list)