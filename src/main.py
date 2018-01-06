# -*- coding: utf-8 -*-
# main.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from plot import Plotter, EmbeddingProjector
from agent import Agent, RandomAgent
from environment import Environment
from parameters import Parameters
from memory import Memory

import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from os.path import exists


OUT_FOLDER = "out"

def train(params_path):
    Plotter.load(OUT_FOLDER)
    Parameters.load(params_path)
    environment = Environment()
    agent = Agent(environment)
    agent.train()

def plot_tsne(params_path):
    Parameters.load(params_path)
    environment = Environment()
    agent = Agent(environment)

    states = list()
    for i in range(20):
        state_t, _, _, _, _, _, _ = agent.memory.bring_back_memories()
        states.append(state_t)
    states = np.concatenate(states, axis=0)

    hidden_repr, v_values = list(), list()
    for state in states:
        hidden_repr.append(agent.tf_session.run(agent.dqn.h_fc1, {agent.dqn_input: [state]}))
        v_values.append(agent.tf_session.run(agent.dqn.q_values, {agent.dqn_input: [state]}).max())
    v_values = np.asarray(v_values)
    hidden_repr = np.asarray(hidden_repr)
    hidden_repr = hidden_repr.reshape(hidden_repr.shape[0], hidden_repr.shape[1] * hidden_repr.shape[2])

    projector = EmbeddingProjector(n_components=2)
    projector.save_plot(hidden_repr, v_values, OUT_FOLDER)

def plot_figures():
    Plotter.load(OUT_FOLDER)
    Plotter.save_plots(OUT_FOLDER)

def plot_conv_layers(params_path):
    Parameters.load(params_path)
    environment = Environment()
    agent = Agent(environment)
    Plotter.plot_conv_layers(agent)


def play_random(params_path):
    Parameters.load(params_path)
    environment = Environment()
    agent = RandomAgent(environment)
    all_scores = agent.play()
    print('mean: ', np.mean(all_scores), '\tstd: ', np.std(all_scores))

def play_pre_trained(params_path):
    Parameters.load(params_path)
    Parameters.DISPLAY = True
    environment = Environment()
    agent = Agent(environment, load_memory=False)
    agent.play()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parameters_json', type=str, help='The path of the parameters to load (! Will be edited when saving!)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Make the agent learn')
    group.add_argument('--plot', action='store_true', help='Plot the results obtained during training')
    group.add_argument('--plot-tsne', action='store_true', help='Plot t-SNE from current checkpoint and current memory')
    group.add_argument('--plot-layers', action='store_true', help='Plot convolutional layers output from current checkpoint')
    group.add_argument('--reset-plot', action='store_true', help='Delete results obtained during training')
    group.add_argument('--random', action='store_true', help='Play 500 games with random action selection and print the mean/std')
    group.add_argument('--play', action='store_true', help='Play the game with pre-trained model')

    args = parser.parse_args()
    assert exists(args.parameters_json)

    if args.train:
        train(args.parameters_json)
    elif args.plot:
        plot_figures()
    elif args.plot_tsne:
        plot_tsne(args.parameters_json)
    elif args.plot_layers:
        plot_conv_layers(args.parameters_json)
    elif args.reset_plot:
        Plotter.reset(OUT_FOLDER)
        Memory.reset(args.parameters_json)
        print("Training results deleted from folder %s and Memory was removed from root." % OUT_FOLDER)
    elif args.random:
        play_random()
    elif args.play:
        play_pre_trained(args.parameters_json)
