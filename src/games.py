# -*- coding: utf-8 -*-
# games.py : Atari games manager
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

import gym


class Games:

    def __init__(self):   

        self.ASTEROIDS = "Asteroids-v0"
        self.PONG = "Pong-v0"
        self.SPACE_INVADERS = "SpaceInvaders-v0"
        self.TENNIS = "Tennis-v0"

        self.available_games_list = [self.ASTEROIDS, self.PONG, self.SPACE_INVADERS, self.TENNIS]
        self.action_space = {}

    def define_action_spaces(self):
        
        for game in self.available_games_list:
            dummy_env = gym.envs.make(game)
            self.action_space[game] = dummy_env.action_space.shape[0]
            # Ensure that the action space is discrete
            # Our deep RL models are based on the assumption that action selection
            # is always binary
            actions = dummy_env.action_space
            assert(type(actions) == gym.spaces.discrete.Discrete)

    def get_action_space(self, game):
        if len(self.action_space.values()) < len(self.available_games_list):
            self.define_action_spaces()
        return(self.action_space[game])

        
