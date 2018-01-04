module load Python  # allow use of python3 command

cd ~/
git clone https://github.com/openai/gym.git
cd gym/
pip3 install -e '.[all]'
pip3 install -e '.[atari]'

pip3 install --user matplotlib opencv-contrib-python  # install necessary packages
cd ~/
mkdir anaconda  # or anything you want
cd ~/anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh  # download Anaconda
bash Anaconda3-5.0.1-Linux-x86_64.sh  # install conda, and follow the instructions while installing
bash  # conda commands path is set in ~/.bashrc, then only bash finds them
conda create -n DQN --clone=/apps/brussel/CO7/magnycours-ib/software/Python/3.6.3-intel-2017b  # copy python environment locally to be allowed to modify it
source activate DQN  # use local environment
conda update conda  # update conda
conda install tensorflow-gpu  # install gpu version of tensorflow

