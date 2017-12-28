# -*- coding: utf-8 -*-
# main.py 
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from plot import Plotter
from agent import Agent
from environment import Environment
from parameters import Parameters

import argparse


def train():
    Plotter.load("out")
    Parameters.load("parameters/dev.json")
    environment = Environment()
    agent = Agent(environment)
    agent.train()

def plot_figures():
    folder = "out/"
    Plotter.load(folder)
    Plotter.save_plots(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Make the agent learn')
    group.add_argument('--plot', action='store_true', help='Plot the results obtained during training')

    args = parser.parse_args()
    if args.train:
        train()
    else:
        plot_figures()
