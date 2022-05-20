import torch
import numpy as np

import simtk.openmm as omm
import simtk.openmm.app as app
import simtk.unit as ommunit

from utils_peptide import convertangularaugmenteddataset

from joblib import Parallel, delayed
import multiprocessing

import os


class MDSimulator:
    def __init__(self, pdbstructure, bGPU=False, sAngularRep='', sOutputpath='', nSimulators=1):

        if sAngularRep in 'ang_auggrouped':
            self.sCoordinateRep = 'AngularAugmented'
        else:
            self.sCoordinateRep = 'Cartesian'

        self.outputPath = sOutputpath
        self.bContraintHBonds = False
        self.bGPU = bGPU
        self.integratorlist = []
        self.simlist = []

        if bGPU:
            platform = omm.Platform.getPlatformByName('CUDA')
        else:
            platform = omm.Platform.getPlatformByName('CPU')

        self.temp = 330.
        self.PDBstructureFile = pdbstructure
        self.forcefieldoptions = '\'amber96.xml\', \'amber96_obc.xml\''
        self.pdb = app.PDBFile(self.PDBstructureFile)
        self.forcefield = app.ForceField(eval(self.forcefieldoptions)[0])

        if self.bContraintHBonds:
            self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff,
                                                         constraints=omm.app.HBonds, soluteDielectric=1.0,
                                                         solventDielectric=80.0)
        else:
            self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff,
                                                         soluteDielectric=1.0,
                                                         solventDielectric=80.0)

        # For parallel processing the forces
        self.createSimObjectList(ommpdb=self.pdb, ommmdsystem=self.mdsystem, nsimulators=3, platform=platform)
        self.atomisticViewNumpy = np.zeros(10)

        self.mdintegrator = omm.LangevinIntegrator(self.temp*ommunit.kelvin, 1/ommunit.picosecond, 0.002*ommunit.picoseconds)

        if bGPU:
            self.simulation = app.Simulation(self.pdb, self.mdsystem, self.mdintegrator, platform)
        else:
            self.simulation = app.Simulation(self.pdb, self.mdsystem, self.mdintegrator)

        # for a prefactor of the learning scheme
        self.tempering_training_iteration = 0.
        self.tempering_training_itermax = 50000.
        self.tempering_check_change_every = 200
        self.tempering_file = os.path.join(sOutputpath, 'a_prefactor.txt')

    def createSimObjectList(self, ommpdb, ommmdsystem, nsimulators, platform):

        self.integratorlist = [omm.LangevinIntegrator(self.temp * ommunit.kelvin, 1 / ommunit.picosecond,
                                                 0.002 * ommunit.picoseconds) for m in range(nsimulators)]

        self.simlist = [app.Simulation(ommpdb, ommmdsystem, self.integratorlist[m], platform) for m in range(nsimulators)]

    def doStep(self):
        self.tempering_training_iteration += 1

    def getLearningPrefactor(self):
        a_init = 0.00001
        a_end = 1.0
        a = 0.
        if self.tempering_training_iteration > self.tempering_training_itermax:
            a = 1.
        elif self.tempering_training_iteration == 0:
            a = a_init
        else:
            a = (1./self.tempering_training_itermax)*self.tempering_training_iteration

        if self.tempering_training_iteration % self.tempering_check_change_every == 0:
            if os.path.isfile(self.tempering_file):
                a = np.loadtxt(self.tempering_file)
        return a

    def getModelType(self):
        return 'MD'

    def getTemperature(self):
        return self.temp

    def getBeta(self):
        return 1./(self.temp*ommunit.kelvin * self.getBoltzmannConstant()) * self.getLearningPrefactor()

    def getBoltzmannConstant(self):
        return ommunit.constants.BOLTZMANN_CONSTANT_kB*ommunit.constants.AVOGADRO_CONSTANT_NA

    def convertToCartesian(self, inputTorch):

        if not self.sCoordinateRep in 'Cartesian':
            lengDim0 = inputTorch.shape[0]
            lengDim1 = inputTorch.shape[1]
            nAtoms = lengDim1 / 5 + 1

            # convert to coordinates x ndatapoints
            if self.bGPU:
                npinput = inputTorch.data.cpu().numpy()
            else:
                npinput = inputTorch.data.numpy()

            # in case we use internally different coordinates than the cartesian representation we need to convert
            # them to cartesian coordinates for openMM
            # TODO CALCULATE JACOBIAN FOR dr/dx and account for it in backward
            npCartesian = convertangularaugmenteddataset(npinput.T, bgrouped=True)
            torchCoordTemp = torch.from_numpy(npCartesian.T)

            if self.bGPU:
                torchCoord = torchCoordTemp.to('cuda')
            else:
                torchCoord = torchCoordTemp
            atomisticView = torchCoord.view([lengDim0, nAtoms, 3])

        else:
            lengDim0 = inputTorch.shape[0]
            lengDim1 = inputTorch.shape[1]
            nAtoms = lengDim1 / 3
            atomisticView = inputTorch.view([lengDim0, nAtoms, 3])

        return atomisticView, nAtoms, lengDim0

    def parallelPotForceEval(self, i, u):
        beta = self.getBeta()

        self.simlist[i].context.setPositions(self.atomisticViewNp[i])
        reporter = self.simlist[i].context.getState(getForces=True, getEnergy=True)
        frep = reporter.getForces(asNumpy=True)
        frep = beta * frep

        upot = reporter.getPotentialEnergy()
        utemp = beta * upot

        return (frep, utemp)


        # for frame in atomisticView.split(1):
        #     sqframe = frame.squeeze()
        #
        #     # convert to numpy for loading into OpenMM
        #     if mdsimulator.bGPU:
        #         npframe = sqframe.data.cpu().numpy()
        #     else:
        #         npframe = sqframe.data.numpy()
        #
        #     mdsimulator.simlist[0].context.setPositions(npframe)
        #     reporter = mdsimulator.simlist[0].context.getState(getForces=True, getEnergy=True)
        #
        #     frep = reporter.getForces(asNumpy=True)
        #     MDLoss.checkNanForces(input=frep)
        #
        #     # make unitless (or just leave the unit of the distance)
        #     frep = beta * frep
        #     # remove the unit from the derivative. Here it would be 1/[x] with [x] the units of the coordinates
        #     if mdsimulator.bGPU:
        #         tempvec = -torch.from_numpy(frep / frep.unit)
        #         dudx[idx, :, :] = tempvec.to('cuda')
        #     else:
        #         dudx[idx, :, :] = -torch.from_numpy(frep / frep.unit)
        #
        #     upot = reporter.getPotentialEnergy()
        #     # make unitless
        #     utemp = beta * upot
        #     u[idx] = -utemp
        #     # u[idx] = upot/upot.unit
        #
        #     idx += 1



class MDLoss(torch.autograd.Function):
    '''
    This class is a loss function log p(x) of Boltzmann distribution.
    '''
    @staticmethod
    def forward(ctx, input, mdsimulator, reduce=False):

        bCPUpar = True

        if MDLoss.checkNanInput(input):
            print input
            #input_prev, u_prev, dudx_prev = ctx.saved_tensors
            #np.savetxt('force_nan_inf.txt', dudx_prev.data.numpy())

        # represent the input vector x in a multiple of 3 for cartesian coordinates
        # each row should relate to the coordinates of an atom.
        #
        atomisticView, nAtoms, lengDim0 = mdsimulator.convertToCartesian(inputTorch=input)

        if mdsimulator.bGPU:
            u = torch.zeros(lengDim0, device=torch.device("cuda"))
            dudx = torch.zeros(lengDim0, nAtoms, 3, device=torch.device("cuda"))
        else:
            u = torch.zeros(lengDim0)
            dudx = torch.zeros(lengDim0, nAtoms, 3)
        idx = 0

        # get beta
        beta = mdsimulator.getBeta()

        # implement parallel loop
        if bCPUpar and not mdsimulator.bGPU:

            atomisticViewNp = atomisticView.data.numpy()
            mdsimulator.atomisticViewNumpy = atomisticViewNp

            num_cores = 2  # multiprocessing.cpu_count()
            res = Parallel(n_jobs=2, require='sharedmem')(delayed(mdsimulator.parallelPotForceEval)(i, u) for i in range(lengDim0))

        else:
            for frame in atomisticView.split(1):
                sqframe = frame.squeeze()

                # convert to numpy for loading into OpenMM
                if mdsimulator.bGPU:
                    npframe = sqframe.data.cpu().numpy()
                else:
                    npframe = sqframe.data.numpy()

                mdsimulator.simlist[0].context.setPositions(npframe)
                reporter = mdsimulator.simlist[0].context.getState(getForces=True, getEnergy=True)

                frep = reporter.getForces(asNumpy=True)
                MDLoss.checkNanForces(input=frep)

                # make unitless (or just leave the unit of the distance)
                frep = beta * frep
                # remove the unit from the derivative. Here it would be 1/[x] with [x] the units of the coordinates
                if mdsimulator.bGPU:
                    tempvec = -torch.from_numpy(frep / frep.unit)
                    dudx[idx, :, :] = tempvec.to('cuda')
                else:
                    dudx[idx, :, :] = -torch.from_numpy(frep/frep.unit)

                upot = reporter.getPotentialEnergy()
                # make unitless
                utemp = beta * upot
                u[idx] = -utemp
                #u[idx] = upot/upot.unit

                idx += 1

        #print 'Reduce {}'.format(reduce)
        if reduce or input.dim() == 1:
            output = u.sum()
        else:
            output = u

        ctx.save_for_backward(input, u, dudx)
        ctx.reduce = reduce
        ctx.bDebug = False
        ctx.nAtoms = nAtoms
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, u, dudx = ctx.saved_tensors
        #grad = input.mul(2.)
        dtypeinput = input.dtype

        grad = -dudx.view([-1, ctx.nAtoms*3])
        grad_input = grad_output*grad

        if ctx.bDebug:
            print 'Grad_output'
            print grad_output
            print 'Grad_input'
            print grad_input

        return grad_input, None, None

    @staticmethod
    def checkNanInput(input):
        nans = torch.isnan(input)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            np.savetxt('nanout.txt', input.data.numpy())
            return True
        else:
            return False

    @staticmethod
    def checkNanForces(input):
        '''
        check if any force is NAN or INF
        :param input:
        :return:
        '''

        # NAN
        boolnanpos = np.isnan(input)
        if boolnanpos.any():
            print 'NaN in returned forces from openMM.'
        #np.savetxt('force_nan_inf.txt', input)
        # INF
        boolinfpos = np.isinf(input)
        if boolinfpos.any():
            print 'INF in returned forces from openMM.'
        #np.savetxt('force_nan_inf.txt', input)
        #input[bool]



import unittest

class TestMDLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(138323)
        self.mdsim = MDSimulator('/home/schoeberl/Dropbox/PhD/projects/2018_07_06_openmm/ala2/ala2_adopted.pdb')
        dat = torch.rand(2, 66)
        self.x = dat
        self.x.requires_grad = True

        self.f = MDLoss.apply
        self.y = self.f(self.x, self.mdsim, True)
        self.z = self.y.pow(2.)

    def test_backward(self):
        # calculate gradient
        self.z.backward(retain_graph=True)
        # print gradient calculation
        print torch.autograd.grad(self.z, self.y, retain_graph=True)
        print torch.autograd.grad(self.y, self.x, retain_graph=True)
        print torch.autograd.grad(self.z, self.x, retain_graph=True)
        # check if gradient has been calculated
        self.assertFalse( self.x.grad is None )

    def test_numerics(self):
        test = torch.autograd.gradcheck(self.f, (self.x, self.mdsim, True), eps=1e-3, atol=1e-3)
        self.assertTrue(True)

    #def test_gradcheck(self):
    #    test = torch.autograd.gradcheck(f, (x, mdsim, True), eps=1e-3, atol=1e-3)