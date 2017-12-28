# -*- coding: utf-8 -*-
# environment.py 
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from history import FramesHistory
from parameters import Parameters
from plot import Plotter
import utils

import gym
from gym.envs.atari.atari_env import AtariEnv

import numpy as np
import matplotlib.pyplot as plt


class Environment:

    SHOW_PLT = False

    def __init__(self):

        self.environment = gym.envs.make(Parameters.GAME)
        self.lives = np.inf
        self.new_game()

        # let's have a look at the first cnn input
        if Environment.SHOW_PLT:
            for screen in self.history.get():
                plt.imshow(screen, cmap='gray')
                plt.show()
        

    def new_game(self):
        """
        Create a new game and return the game variables
        """

        self._screen = self.environment.reset()
        first_action = 0
        self.take_action(first_action)
        self.render()
        self.reward = 0
        self.episode_score = 0
        self.terminal = False

        self.initialize_screens_history()

        return(self.screen, first_action, self.reward, self.terminal)
    

    def initialize_screens_history(self):

        """ initialize the history by pushing the first screen 4 (AGENT_HISTORY_LENGTH) times """

        self.history = FramesHistory()
        for _ in range(Parameters.AGENT_HISTORY_LENGTH):
            self.history.add_frame(self.screen)

    
    def add_current_screen_to_history(self):
        self.history.add_frame(self.screen)

    
    def get_input(self):
        return(self.history.get())


    def select_random_action(self):
        return(self.environment.action_space.sample())
    

    def select_smart_action(self):
        """
        Not smart yet
        """
        return(self.select_random_action())


    def take_action(self, action):
        self._previous_screen = self._screen
        self._screen, self.reward, self.terminal, info = self.environment.step(action)
        self.lives = info["ale.lives"]


    
    def process_step(self, action):

        """
        Take the action a certain number of times (Parameters.FRAME_SKIPPING)
        as described in the article
        Return the environment state after taking the action x times
        """
        
        lives_before_action = self.lives
        cumulated_reward = Parameters.NO_REWARD

        # frame skipping (see Parameters for more information)
        skipped = 0 # break is evil
        while(skipped < Parameters.FRAME_SKIPPING and not self.terminal):
            
            self.take_action(action)
            cumulated_reward += self.reward
            self.episode_score += self.reward

            """
            [Article] For games where there is a life counter, the Atari
            2600 emulator also sends the number of lives left in the game, which is then used to
            mark the end of an episode during training.
            """
            if self.lives < lives_before_action:
                cumulated_reward += Parameters.NEGATIVE_REWARD
                self.terminal = True

                Plotter.add_episode_score(self.episode_score)
                self.episode_score = 0
            
            skipped += 1

        self.reward = cumulated_reward

        return(self.state, self.reward, self.terminal)
    
    
    @property
    def screen(self):
        return(utils.preprocess_img(self._previous_screen, self._screen))

    
    @property
    def state(self):
        return(self.screen)

    
    def render(self, mode="human"):
        """
        Display the game on screen only if the display parameter is True
        """
        if(Parameters.DISPLAY):
            self.environment.render(mode=mode)


    def reset(self):
        """
        Reset the environment only if the number of lives is
        """
        if(not self.lives):
            self.environment.reset()

            
