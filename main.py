# -*- coding: utf-8 -*-
# main.py 
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from agent import Agent
from environment import Environment
from parameters import Parameters

import gym
import time
import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    Parameters.load("parameters.json")
    with tf.Session() as session:
        environment = Environment()
        agent = Agent(environment)

        while True:
            environment.render(mode='human')
            for t in range(Parameters.MAX_STEPS):
                a = agent.select_action()
                
                s_prime, r, done = environment.process_step(a)
                if done:
                    break # End of episode
                time.sleep(1.0 / Parameters.FPS) # Wait one step
                environment.render()

            s = s_prime
