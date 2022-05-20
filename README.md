# Embedded_Physics_MD
Embedded-physics machine learning for coarse-graining and collective variable discovery without data



Use results/command.sh to generate the simple 2d-case in the paper.

The installation is partly based on an anaconda python environment. I provide the environment specifics as spec-file.txt (anaconda) or in requirements_pip.txt (pip). Only the cpu version of torch is used. It should be easy to change this by installing the corresponding gpu version of pytorch 1.1 and setting the flag --gpu_mode 1 in the command below. To install use the following bash shell commands:

# latex and dvipng is required
apt-get install latex dvipng anaconda
# create conda env
conda create --name embph python=3.6 --file spec-file.txt
# activate env
conda activate embph
# run the test
python main.py --dataset quad --model_type VARjoint --epoch 50000 --z_dim 1 --seed 3251 --AEVB 1 --samples_per_mean 4 --sharedlogvar 1 --outputfreq 20 --samples_pred 4000 --sharedlogvar 0 --sharedencoderlogvar 0 --gpu_mode 0 --batch_size 4000 --stepSched 500
