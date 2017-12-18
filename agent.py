# -*- coding: utf-8 -*-
# agent.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from environment import Environment
from dqn import DQN

import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self):
        
        self.environment = Environment()
        self.dqn = DQN(self.environment.get_input())


    def select_action(self, s):
        # The agent uses its experience and expertise acquired
        # through deep learning to make intelligent actions
        return np.random.randint(0, self.n_actions, size=1)[0]