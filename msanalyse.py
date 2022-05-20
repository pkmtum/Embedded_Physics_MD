# THIS IS A TEST SCRIPT

import argparse, os
import numpy as np

import matplotlib.pyplot as plt

class StepStore:
    def __init__(self, totsteps):
        self.totalsteps = totsteps
        self.storelist = []
        for i in range(self.totalsteps):
            self.storelist.append(np.zeros([10, 2]))

    def storestep(self, step, radpottemp):
        self.storelist[step] = np.copy(radpottemp)


def findaffectedpaths(path, sfilter='*'):
    import fnmatch

    dirsep = path
    dirlist = fnmatch.filter(os.listdir(dirsep), sfilter)
    pathlist = list()
    for s in dirlist:
        pathlist.append(dirsep + '/' + s)
    return pathlist


def analysestep(path='', stepstoreobj=None):
    data = np.genfromtxt(os.path.join(path, 'CG-CG.pot.new'), dtype=None , delimiter=" ")
    stepsrad = len(data)
    radpot = np.zeros([stepsrad, 2])
    counter = 0

    psplit = path.split('_')
    stepnumber = int(psplit[-1])

    for datitem in data:
        radpot[counter, 0] = datitem[0]
        radpot[counter, 1] = datitem[1]
        counter += 1

    stepstoreobj.storestep(step=stepnumber, radpottemp=radpot)


def plotCGpotential(stepstorage=None):

    f, ax = plt.subplots(1)
    ax.grid(ls='dashed')

    for stepitem in stepstorage.storelist:
        ax.plot(stepitem[:, 0], stepitem[:, 1])

    #ax.set_axisbelow(True)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$U$ ')
    ax.set_ylim([-5, 10])
    #ax.legend()
    f.savefig('u_iteration.pdf', bbox_inches='tight')

def analysesteps(args):
    # find the completed steps:
    pathsteplist = findaffectedpaths(path=args.path, sfilter='*step_*')

    stepstorage = StepStore(totsteps=len(pathsteplist))

    for steppath in pathsteplist:
        analysestep(path=steppath, stepstoreobj=stepstorage)

    print "Storing process done."
    return stepstorage


def parse_args():
    desc = "Analysis tool for MS SS 2018."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--path', type=str, default=os.getcwd(),
                        help='Provide the working directory where step folders are placed.')#, required=True)
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # analyse the steps
    stepstorage = analysesteps(args)

    # plot the CG potential
    plotCGpotential(stepstorage=stepstorage)

if __name__ == '__main__':
    main()