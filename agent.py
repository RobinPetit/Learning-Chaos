# -*- coding: utf-8 -*-
# agent.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from utils import reward_clipper
from environment import Environment
from memory import Memory
from parameters import Parameters
from dqn import DQN
from os import path, makedirs

import random
import time

import numpy as np
import tensorflow as tf


class Agent:

    def __init__(self, environment):
        
        self.environment = environment
        self.memory = Memory()
        self.step = 0
        
        # initialize the DQN and target DQN (with respective placeholders)
        self.dqn_input = tf.placeholder(tf.float32, [None, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH], name = "DQN_input")
        self.dqn = DQN(self.dqn_input)

        self.target_dqn_input = tf.placeholder(tf.float32, [None, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH], name = "target_DQN_input")
        self.target_dqn = DQN(self.target_dqn_input)

        # initialize the tensorflow session and variables
        self.tf_session = tf.Session()
        load_session()
        
    
    def load_session(self):
        save_file = path.join(Parameters.SESSION_SAVE_DIRECTORY, Parameters.SESSION_SAVE_FILENAME)
        if path.exists(save_file) and path.isfile(save_file):
                # restore from a previously saved session
                tf_saver = tf.train.Saver()
                tf_saver.restore(self.tf_session, save_file)
        else:
                # initialize from scratch
                self.tf_session.run(tf.global_variables_initializer())
                
    
    def save_session(self):
        save_file = path.join(Parameters.SESSION_SAVE_DIRECTORY, Parameters.SESSION_SAVE_FILENAME)
        if not path.exists(save_file):
                makedirs(Parameters.SESSION_SAVE_DIRECTORY)
        tf_saver = tf.train.Saver()
        tf_saver.save(self.tf_session, save_file)


    def train(self):

        self.step = 0

        while(self.step < Parameters.MAX_STEPS):

            self.environment.render(mode='human')
            self.environment.terminal = False
            for t in range(Parameters.MAX_STEPS):
                
                # select an action
                action = self.select_action()
                
                # process step
                state, reward, terminal = self.environment.process_step(action)
                
                # observe the consequence of the action
                self.observe(state, action, reward, terminal)

                if terminal:
                    self.environment.reset()
                    break

                if(Parameters.SLEEP_BETWEEN_STEPS):
                    time.sleep(1.0 / Parameters.FPS) # Wait one step

                self.environment.render()
                
                self.step += 1

            

    
    def batch_q_learning(self):
        """
        Apply Q-learning updates, or minibatch updates, to samples of experience,
        (s, a, r, s') ~ U(D), drawn at random from the pool of stored samples.
        """

        if(self.memory.get_usage() > Parameters.AGENT_HISTORY_LENGTH):
            
            state_t, action, reward, state_t_plus_1, terminal = self.memory.bring_back_memories()

            q_t_plus_1 = self.tf_session.run(self.target_dqn.q_values, {self.target_dqn_input: state_t_plus_1})
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

            target_q_t = (1. - terminal) * Parameters.DISCOUNT_FACTOR * max_q_t_plus_1 + reward

            _, q_t, loss = self.tf_session.run([self.dqn.optimize, self.dqn.q_values, self.dqn.error],
            {
                self.dqn.target_q : target_q_t,
                self.dqn.action : action,
                self.dqn_input : state_t
            })


    def observe(self, screen, action, reward, terminal):
        """
        [Article] The agent observes an image from the emulator, 
        which is a vector of pixel values representing the current screen. 
        In addition it receives a reward representing the change in game score. 

        Updates the environment's history and agent's memory and performs an SGD update
        and/or updates the Target DQN 
        """
        
        # update agent's memory and environment's history
        self.environment.add_current_screen_to_history()
        self.memory.add(screen, action, reward_clipper(reward), terminal)

        # if we started learning
        if(self.step > Parameters.REPLAY_START_SIZE):

            # Perform SGD updates at frequency [Parameters.UPDATE_FREQUENCY]
            if(not(self.step % Parameters.UPDATE_FREQUENCY)):
                self.batch_q_learning()
            
            # Update Target DQN at frequency [Parameters.TARGET_NETWORK_UPDATE_FREQUENCY]
            if(not(self.step % Parameters.TARGET_NETWORK_UPDATE_FREQUENCY)):
                self.update_target_dqn()
            


    def select_action(self):
        """
        The agent uses its experience and expertise acquired
        through deep learning to make intelligent actions (sometimes)
        """

        # compute epsilon at step t
        dt_final = Parameters.INITIAL_EXPLORATION - Parameters.FINAL_EXPLORATION
        dt = self.step - Parameters.REPLAY_START_SIZE
        df = Parameters.FINAL_EXPLORATION_FRAME - Parameters.REPLAY_START_SIZE
        eps = Parameters.INITIAL_EXPLORATION - ((dt / df) * (Parameters.INITIAL_EXPLORATION - Parameters.FINAL_EXPLORATION))
        eps = 0
        if random.random() < eps:
            # take a random action
            action = np.random.randint(0, Parameters.ACTION_SPACE, size=1)[0]
        else:
            # take a smart action
            input_shape = (1, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH)
            dqn_input = self.environment.get_input().reshape(input_shape)
            q_print = self.tf_session.run(self.dqn.q_values, {self.dqn_input: dqn_input})
            print("Q-values: ", q_print, "  at step : ", self.step)
            action = self.tf_session.run(self.dqn.smartest_action, {self.dqn_input: dqn_input})
        
        return(action)


    def update_target_dqn(self):
        """
        Update the target DQN with the value of the DQN
        This method is called at a frequency C (from Algorithm 1 in the [Article])
        """

        for learning_parameter in self.dqn.learning_parameters:
            dqn_value = self.dqn.get_value(learning_parameter, self.tf_session)
            if(dqn_value is not None):
                self.target_dqn.set_value(learning_parameter, dqn_value, self.tf_session)
            else:
                print("Impossible to set value: None")

    
    def play(self):
        print("To do")

    

        
