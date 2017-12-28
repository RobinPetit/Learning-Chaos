# -*- coding: utf-8 -*-
# main.py 
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from agent import Agent
from environment import Environment
from parameters import Parameters


if __name__ == "__main__":
    
    Parameters.load("parameters/dev.json")
    environment = Environment()
    agent = Agent(environment)
    
    agent.train()

    
