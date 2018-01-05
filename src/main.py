# -*- coding: utf-8 -*-
# main.py
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from plot import Plotter
from agent import Agent, RandomAgent
from environment import Environment
from parameters import Parameters
from memory import Memory

import argparse

import numpy as np

from os.path import exists

OUT_FOLDER = "out"

def train(params_path):
    Plotter.load(OUT_FOLDER)
    Parameters.load(params_path)
    environment = Environment()
    agent = Agent(environment)
    agent.train()

def plot_figures():
    Plotter.load(OUT_FOLDER)
    Plotter.save_plots(OUT_FOLDER)

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
    group.add_argument('--reset-plot', action='store_true', help='Delete results obtained during training')
    group.add_argument('--random', action='store_true', help='Play 500 games with random action selection and print the mean/std')
    group.add_argument('--play', action='store_true', help='Play the game with pre-trained model')

    args = parser.parse_args()
    assert exists(args.parameters_json)

    if args.train:
        train(args.parameters_json)
    elif args.plot:
        plot_figures()
    elif args.reset_plot:
        Plotter.reset(OUT_FOLDER)
        Memory.reset(args.parameters_json)
        print("Training results deleted from folder %s and Memory was removed from root." % OUT_FOLDER)
    elif args.random:
        play_random()
    elif args.play:
        play_pre_trained(args.parameters_json)
