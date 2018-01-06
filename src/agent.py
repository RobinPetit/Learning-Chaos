# -*- coding: utf-8 -*-
# agent.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from utils import reward_clipper
from environment import Environment
from memory import Memory, PrioritizedMemory, BalancedMemory
from parameters import Parameters
from plot import Plotter
from dqn import DQN
from dddqn import DDDQN

from os import path, makedirs
from datetime import datetime, timedelta
import random
import time
import sys
import numpy as np
import tensorflow as tf

from random import randint  as py_randint  # faster than np for a single element. np is
                                           # interesting to generate many numbers at once

randint = lambda a, b: py_randint(a, b-1)  # to emulate np randint behaviour: return in [a, b)


class RandomAgent:
    def __init__(self, environment):
        self.action_space = Parameters.GAMES.get_action_space(Parameters.GAME)
        self.environment = environment

    def play(self, nb_simulations=500):
        all_scores = list()
        score = 0
        for sim in range(nb_simulations):
            sys.stdout.write('\r{} out of {}'.format(sim+1, nb_simulations))
            self.environment.reset()
            self.environment.terminal = False
            while self.environment.get_lives() > 0:
                action = randint(0, self.action_space)
                _, reward, done = self.environment.process_step(action)
                score += reward
                if done:
                    self.environment.terminal = False
            all_scores.append(score)
            score = 0
        print('')
        return all_scores


class Agent:

    def __init__(self, environment, load_memory=True):

        self.action_space = Parameters.GAMES.get_action_space(Parameters.GAME)
        self.environment = environment
        if load_memory:
            self.memory = PrioritizedMemory() if Parameters.USE_PRIORITIZATION else Memory()
        self.step = 0

        # select the type of DQN based on Parameters
        dqn_type = DDDQN if Parameters.USE_DDDQN else DQN


        # initialize the DQN and target DQN (with respective placeholders)
        self.dqn_input = tf.placeholder(tf.float32, [None, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH], name = "DQN_input")
        self.dqn = dqn_type(self.dqn_input, self.action_space)

        self.target_dqn_input = tf.placeholder(tf.float32, [None, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH], name = "target_DQN_input")
        self.target_dqn = dqn_type(self.target_dqn_input, self.action_space)

        # initialize the tensorflow session and variables
        self.tf_session = tf.Session()
        self.tf_saver = tf.train.Saver()
        self.load_session()
        self.initial_time = self.last_time = time.time()
        self.initial_step = self.step
        self.last_action = randint(0, self.action_space)

    def load_session(self):
        save_file = path.join(Parameters.SESSION_SAVE_DIRECTORY, Parameters.SESSION_SAVE_FILENAME)
        if path.exists(Parameters.SESSION_SAVE_DIRECTORY):
            # restore from a previously saved session
            print("Loading session from", save_file)
            self.tf_saver.restore(self.tf_session, save_file)
            self.step = Parameters.CURRENT_STEP
        else:
            # initialize from scratch
            print("Loading new session")
            self.tf_session.run(tf.global_variables_initializer())

    def save_session(self):
        time_at_start_save = time.time()
        sys.stdout.write('{}: [Step {}k  --  Took {:3.2f} s] '.format(datetime.now(), self.step//1000, time_at_start_save - self.last_time))
        self.last_time = time_at_start_save

        Parameters.CURRENT_STEP = self.step
        a = time.time()
        Parameters.update()
        b = time.time()
        sys.stdout.write('[{:3.2f}s for json] '.format(b-a))

        save_file = path.join(Parameters.SESSION_SAVE_DIRECTORY, Parameters.SESSION_SAVE_FILENAME)
        if not path.exists(Parameters.SESSION_SAVE_DIRECTORY):
                makedirs(Parameters.SESSION_SAVE_DIRECTORY)
        a = time.time()
        self.tf_saver.save(self.tf_session, save_file)
        b = time.time()
        sys.stdout.write('[{:3.2f}s for tf] '.format(b-a))

        a = time.time()
        Plotter.save("out")
        b = time.time()
        sys.stdout.write('[{:3.2f}s for Plotter] '.format(b-a))
        a = time.time()
        self.memory.save_memory()
        b = time.time()
        sys.stdout.write('[{:3.2f}s for memory] '.format(b-a))
        post_save_time = time.time()
        sys.stdout.write('[Required {:3.2f}s to save all] '.format(post_save_time-time_at_start_save))
        self.last_time = post_save_time
        elapsed_time = time.time() - self.initial_time
        remaining_seconds = elapsed_time*(Parameters.MAX_STEPS - self.step)/(self.step - self.initial_step)
        print("eta: {}s".format((timedelta(seconds=remaining_seconds))))

    def train(self):

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

                if Parameters.DISPLAY:
                    time.sleep(1.0 / Parameters.FPS) # Wait one step

                self.environment.render()

                self.step += 1

                if self.step % 10000 == 0:
                    self.save_session()
                if self.step % Parameters.SHORT_TERM_MEMORY_UPDATE_PERIOD == 0:  # This name is waaaaaaaaaaaaaaaay too long <3
                    self.memory.update_short_term()

                if terminal:
                    self.environment.reset()
                    break


        self.save_session()

    def batch_q_learning(self):
        """
        Apply Q-learning updates, or minibatch updates, to samples of experience,
        (s, a, r, s') ~ U(D), drawn at random from the pool of stored samples.
        """

        if(self.memory.get_usage() > Parameters.AGENT_HISTORY_LENGTH):

            state_t, action, reward, state_t_plus_1, terminal, i_s_weights, memory_indices = self.memory.bring_back_memories()

            q_t_plus_1 = self.tf_session.run(self.target_dqn.q_values, {self.target_dqn_input: state_t_plus_1})
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

            target_q_t = (1. - terminal) * Parameters.DISCOUNT_FACTOR * max_q_t_plus_1 + reward

            _, q_t, losses = self.tf_session.run([self.dqn.optimize, self.dqn.q_values, self.dqn.errors],
            {
                self.dqn.target_q : target_q_t,
                self.dqn.action : action,
                self.dqn_input : state_t,
                self.dqn.i_s_weights : i_s_weights
            })

            self.memory.update(memory_indices, np.squeeze(q_t), losses, self.get_learning_completion())
            input_shape = (1, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH)
            dqn_input = self.environment.get_input().reshape(input_shape)
            q_values = self.tf_session.run(self.dqn.q_values, {self.dqn_input: dqn_input})
            Plotter.add_q_values_at_t(q_values)
        else:
            print('[WARNING] Not enough memory for a batch')

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
        self.memory.add(screen, action, reward_clipper(reward), terminal,)

        # if we started learning
        if(self.step > Parameters.REPLAY_START_SIZE and self.step % Parameters.FRAME_SKIPPING == 0):
            nb_selected_actions = self.step // Parameters.FRAME_SKIPPING
            # Perform SGD updates at frequency [Parameters.UPDATE_FREQUENCY]
            if(not(nb_selected_actions % Parameters.UPDATE_FREQUENCY)):
                self.batch_q_learning()

            # Update Target DQN at frequency [Parameters.TARGET_NETWORK_UPDATE_FREQUENCY]
            if(not(nb_selected_actions % Parameters.TARGET_NETWORK_UPDATE_FREQUENCY)):
                self.update_target_dqn()

    def get_learning_completion(self):
        """
        Returns the number of performed learning steps divided by the maximum number of steps
        """
        return min(1.0, self.step / Parameters.FINAL_EXPLORATION)

    def select_action(self, eps=None):
        """
        The agent uses its experience and expertise acquired
        through deep learning to make intelligent actions (sometimes)
        """
        if self.step % Parameters.FRAME_SKIPPING != 0:
            return self.last_action
        # compute epsilon at step t
        completion = self.get_learning_completion()
        if eps is None:
            eps = Parameters.INITIAL_EXPLORATION - (completion * (Parameters.INITIAL_EXPLORATION - Parameters.FINAL_EXPLORATION))
        if random.random() < eps:
            # take a random action
            action = randint(0, self.action_space)
        else:
            # take a smart action
            input_shape = (1, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH)
            dqn_input = self.environment.get_input().reshape(input_shape)
            action = self.tf_session.run(self.dqn.smartest_action, {self.dqn_input: dqn_input})

        self.last_action = action
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
        self.environment.render(mode='human')
        while True:
            self.environment.reset()
            game_reward = 0
            while True:
                action = self.select_action(eps=.1)
                _, reward, done = self.environment.process_step(action)
                game_reward += reward
                if done:
                    self.environment.terminal = False
                    game_reward += 1
                    if self.environment.get_lives() == 0:
                        print('Score:', game_reward)
                        game_reward = 0
                        break
                    self.environment.reset()
                time.sleep(1. / Parameters.FPS)
                self.environment.render()

