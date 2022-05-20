# -*- coding: utf-8 -*
from matplotlib import rc
from matplotlib import rcParams


def loadfontsetting():

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    font = {'family': 'normal',
            # 'weight' : 'bold',
            'size': 16}
    rc('font', **font)
    rc('text', usetex=True)
    leg = {'fontsize': 18}  # ,
    # 'legend.handlelength': 2}
    #rc('legend', **leg)
    rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]