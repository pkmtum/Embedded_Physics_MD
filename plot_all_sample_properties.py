import os
import numpy as np


iterations = np.arange(0, 21401, 100)
iterations[0] = 1

list_file_name_samples = ['samples_aevb_' + str(i) for i in iterations]
print list_file_name_samples

referenceDirectory = '/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/prediction/propteinpropcal/'

for file_name in list_file_name_samples:
    str_command = 'python estimate_properties_o.py --fileNamePred ' + file_name + ' --referenceDirectory ' + referenceDirectory
    print str_command
    os.system(str_command)
