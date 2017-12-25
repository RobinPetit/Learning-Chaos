# Dependencies

## Libraries

* numpy
* scipy
* matplotlib
* gym + ALE
* tensorflow
* scikit-learn
* openCV

## Install ALE

If you have Anaconda installed, first do:

```{r, engine='bash', count_lines}
conda install libgcc
```

To install gym with Atari_py:

```{r, engine='bash', count_lines}
# Install dependencies
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

# Install gym and atari_py rom repository
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
pip install -e '.[atari]'

# Install tensorflow (assuming you have Anaconda installed)
conda install tensorflow
# For GPU version, check https://www.tensorflow.org/install/install_linux
```

Mot gentil pour Robin: <3
