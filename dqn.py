# -*- coding: utf-8 -*-
# dqn.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from utils import Params

import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, session, n_actions):
        self.session = session # Tensorflow session
        self.n_actions = n_actions
        # TODO: init neural network

    def select_action(self, s):
        # The agent uses its experience and expertise acquired
        # through deep learning to make intelligent actions
        return np.random.randint(0, self.n_actions, size=1)[0]
