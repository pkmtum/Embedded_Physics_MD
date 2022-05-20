
import numpy as np
import os
import argparse

def getprefixcrc(jobname='test', bgpu=False, email=False):
    sprefix = ''
    sprefix = sprefix + '#!/bin/bash\n'
    if email:
        sprefix = sprefix + '#$ -M mschoebe@nd.edu\n' # Email address for job notification
        # sprefix = sprefix + '#$ -m bea\n' # Email address for job notification
    sprefix = sprefix + '#$ -N ' + jobname + '\n' # job name
    if bgpu:
        sprefix = sprefix + '#$ -q gpu-debug\n' # queue
    else:
        sprefix = sprefix + '#$ -pe smp 4\n' # core size
        #sprefix = sprefix + '#$ -pe mpi-24 4\n' # core size
        sprefix = sprefix + '#$ -q long\n'  # queue
    sprefix = sprefix + '\n'  # space
    sprefix = sprefix + 'module load cuda\n'  # queue
    if bgpu:
        sprefix = sprefix + 'source activate anacPyTorchCuda\n'  # queue
    else:
        sprefix = sprefix + 'source activate anacPyTorch\n'  # queue
    sprefix = sprefix + '\n'
    return sprefix

def startanalysisSingle():
    import fnmatch
    dirlist = fnmatch.filter(os.listdir(os.getcwd()), 'b_*c_*z_*')

    command = ''
    command = command + '#!/bin/bash\n'

    for s in dirlist:
        filenamepred = s + '/samples'
        command = command + 'python estimate_properties.py --fileNamePred '
        command = command + 'samples'
        command = command + ' --predFilePath ' + os.getcwd() + '/' + s + '/'
        command = command + ' --cluster 0\n'

    # create file
    f = open('estimateprop.sh', 'w')
    f.write(command)
    f.close()

def findaffectedpaths():
    import fnmatch

    dirsep = os.getcwd() + '/results/separate_1000'
    print dirsep
    dirlist = fnmatch.filter(os.listdir(dirsep), 'sep_b_*c_*z_*')
    print dirlist
    pathlist = list()
    for s in dirlist:
        pathlist.append(dirsep + '/' + s)
    return pathlist

def commandforestimateproperties(conf='m', prefix='samples', path=''):
    cmd = ''
    cmd = cmd + 'python estimate_properties.py --fileNamePred '
    cmd = cmd + prefix
    cmd = cmd + ' --predFilePath ' + path + '/'
    cmd = cmd + ' --conformation ' + conf
    cmd = cmd + ' --cluster 0\n'

    return cmd


def startanalysisSep():
    dirlist = findaffectedpaths()
    print dirlist

    command = ''
    command = command + '#!/bin/bash\n'

    for s in dirlist:
        command = command + commandforestimateproperties(conf='m', prefix='samples', path=s)
        command = command + commandforestimateproperties(conf='a', prefix='samples_a', path=s)
        command = command + commandforestimateproperties(conf='b1', prefix='samples_b1', path=s)
        command = command + commandforestimateproperties(conf='b2', prefix='samples_b2', path=s)

    # create file
    f = open('estimateprop.sh', 'w')
    f.write(command)
    f.close()

"""main"""
def main(type='mixed'):
    #batch_size = np.array([16, 32, 64, 128, 246, 512]).astype(int)
    batch_size = np.array([16, 32, 64, 512, 1000]).astype(int)
    c = np.array([0.1, 0.01])
    z_dim = np.array([2, 5, 10]).astype(int)
    bgpu = True
    epoch = 2000
    bangdata = False
    nsamplespred = 4000

    for b_size in np.nditer(batch_size):
        for clipping in np.nditer(c):
            for z in np.nditer(z_dim):
                sPostFix = type + '_b_' + str(b_size) + '_c_' + str(clipping) + '_z_' + str(z)
                if type == 'mixed':
                    spythoncommand = 'python main.py --dataset m_10437 --gan_type WGAN_peptide'
                    spythoncommand = spythoncommand + ' --epoch ' + str(int(epoch)) + ' --gpu_mode ' + str(int(bgpu)) + ' --clusterND 1'
                    spythoncommand = spythoncommand + ' --batch_size ' + str(b_size)
                    spythoncommand = spythoncommand + ' --clipping ' + str(clipping)
                    spythoncommand = spythoncommand + ' --z_dim ' + str(z)
                    spythoncommand = spythoncommand + ' --outPostFix ' + sPostFix
                else:
                    spythoncommand = 'python main.py --dataset a_1000 --gan_type WGAN_peptide'
                    spythoncommand = spythoncommand + ' --epoch ' + str(int(epoch)) + ' --gpu_mode ' + str(int(bgpu)) + ' --clusterND 1'
                    spythoncommand = spythoncommand + ' --batch_size ' + str(b_size)
                    spythoncommand = spythoncommand + ' --clipping ' + str(clipping)
                    spythoncommand = spythoncommand + ' --z_dim ' + str(z)
                    spythoncommand = spythoncommand + ' --useangulardat ' + str(int(bangdata))
                    spythoncommand = spythoncommand + ' --outPostFix ' + sPostFix
                    spythoncommand = spythoncommand + ' --samples_pred ' + str(nsamplespred)
                    spythoncommand = spythoncommand + '\n'
                    spythoncommand = spythoncommand + 'python main.py --dataset b1_1000 --gan_type WGAN_peptide'
                    spythoncommand = spythoncommand + ' --epoch ' + str(int(epoch)) + ' --gpu_mode ' + str(int(bgpu)) + ' --clusterND 1'
                    spythoncommand = spythoncommand + ' --batch_size ' + str(b_size)
                    spythoncommand = spythoncommand + ' --clipping ' + str(clipping)
                    spythoncommand = spythoncommand + ' --z_dim ' + str(z)
                    spythoncommand = spythoncommand + ' --useangulardat ' + str(int(bangdata))
                    spythoncommand = spythoncommand + ' --samples_pred ' + str(nsamplespred)
                    spythoncommand = spythoncommand + ' --outPostFix ' + sPostFix
                    spythoncommand = spythoncommand + '\n'
                    spythoncommand = spythoncommand + 'python main.py --dataset b2_1000 --gan_type WGAN_peptide'
                    spythoncommand = spythoncommand + ' --epoch ' + str(int(epoch)) + ' --gpu_mode ' + str(int(bgpu)) + ' --clusterND 1'
                    spythoncommand = spythoncommand + ' --batch_size ' + str(b_size)
                    spythoncommand = spythoncommand + ' --clipping ' + str(clipping)
                    spythoncommand = spythoncommand + ' --z_dim ' + str(z)
                    spythoncommand = spythoncommand + ' --useangulardat ' + str(int(bangdata))
                    spythoncommand = spythoncommand + ' --samples_pred ' + str(nsamplespred)
                    spythoncommand = spythoncommand + ' --outPostFix ' + sPostFix

                prefixcrc = getprefixcrc(jobname=sPostFix, bgpu=bgpu)
                filecontent = prefixcrc + spythoncommand

                # create file
                f = open(sPostFix + '.sh', 'w')
                f.write(filecontent)
                f.close()

                command = 'qsub ' + sPostFix + '.sh'
                os.system(command)


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--type', type=str, default='qsub',
                        choices=['analysesep', 'analysesing', 'qsubm', 'qsubs'],
                        help='analyse or qsub the jobs', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # declare instance for GAN
    if args.type == 'analysesep':
        startanalysisSep()
    elif args.type == 'analysesing':
        startanalysisSingle()
    elif args.type == 'qsubm':
        main(type='mixed')
    elif args.type == 'qsubs':
        main(type='sep')