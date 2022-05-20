import numpy as np
import os


def findaffectedpaths():
    import fnmatch

    dirsep = os.getcwd() + '/results/separate_1000'
    dirlist = fnmatch.filter(os.listdir(dirsep), 'sep_b_*c_*z_*')
    pathlist = list()
    for s in dirlist:
        pathlist.append(dirsep + '/' + s)
    return pathlist


def combinesamples(path):
    nPredSamples = 2000

    pMixing = np.array([0.01897807913, 0.6103373091, 0.3706846118])
    pMixing = pMixing/pMixing.sum()

    nSelect = pMixing * nPredSamples
    nSelect = np.round(nSelect)

    # load the configurations:
    confList= list()
    confList.append('samples_a')
    confList.append('samples_b1')
    confList.append('samples_b2')

    # sample randomly the configurations from the various conformations
    predList = list()
    predArray = list()
    indList = list()
    predListSel = list()

    for i in range(0, len(confList)):
        predList.append(path + '/' + confList[i] + '.txt')
        predArray.append(np.loadtxt(predList[i]))
        indList.append(np.random.randint(low=0, high=predArray[i].shape[0], size=int(nSelect[i])))
        predListSel.append(predArray[i][:, indList[i]])

    # Select from predictions the randomly selected indices
    pred = np.zeros([predArray[0].shape[0], int(nSelect.sum())])
    icount = 0
    for i in range(0, len(confList)):
        pred[:, icount:icount+int(nSelect[i])] = predListSel[i]
        icount = icount + int(nSelect[i])

    np.savetxt(path + '/samples.txt', pred)

"""main"""
def main():
    pathlist = findaffectedpaths()

    print pathlist
    for path in pathlist:
        combinesamples(path=path)

    quit()

"""main"""
def main_singlefolder():
    workDir = os.getcwd()
    subPath = '/WGAN_peptide/'
    nPredSamples = 1500

    pMixing = np.array([0.01897807913, 0.6103373091, 0.3706846118])
    pMixing = pMixing/pMixing.sum()

    nSelect = pMixing * nPredSamples
    nSelect = np.round(nSelect)

    # load the configurations:
    confList= list()
    confList.append('a_1000')
    confList.append('b1_1000')
    confList.append('b2_1000')

    # sample randomly the configurations from the various conformations


    predList = list()
    predArray = list()
    indList = list()
    predListSel = list()

    for i in range(0, len(confList)):
        predList.append(workDir + '/results/' + confList[i] + subPath + 'samples.txt')
        predArray.append(np.loadtxt(predList[i]).T)
        indList.append(np.random.randint(low=0, high=predArray[i].shape[0], size=int(nSelect[i])))
        predListSel.append(predArray[i][:, indList[i]])

    # Select from predictions the randomly selected indices
    pred = np.zeros([predArray[0].shape[0], int(nSelect.sum())])
    icount = 0
    for i in range(0, len(confList)):
        pred[:, icount:icount+int(nSelect[i])] = predListSel[i]
        icount = icount + int(nSelect[i])

    np.savetxt(workDir+'/prediction_ab1b2.txt', pred)
    quit()

if __name__ == '__main__':
    main()
