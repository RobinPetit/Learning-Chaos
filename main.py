# -*- coding: utf-8 -*-
# main.py 
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from dqn import Agent
from utils import Params

import gym
import time
import numpy as np
import tensorflow as tf


# Cool (easy) Atari environments
ASTEROIDS = "Asteroids-v0"
PONG = "Pong-v0"
SPACE_INVADERS = "SpaceInvaders-v0"
TENNIS = "Tennis-v0"


if __name__ == "__main__":
    with tf.Session() as session:
        environment = gym.envs.make(SPACE_INVADERS)
        actions = environment.action_space
        assert(type(actions) == gym.spaces.discrete.Discrete)
        agent = Agent(session, actions.shape[0])

        while True:
            s = environment.reset()
            environment.render(mode='human')
            for t in range(Params.MAX_STEPS):
                a = agent.select_action(s)
                
                s_prime, r, done, info = environment.step(a)
                if done:
                    break # End of episode
                time.sleep(1.0/Params.FPS) # Wait one step
                environment.render()

            s = s_prime
