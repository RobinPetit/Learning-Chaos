# -*- coding: utf-8 -*-
# parameters.py : Load parameters from json config file
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from games import Games

import json


class Parameters:

    # environment settings
    GAMES = Games()
    GAME = GAMES.SPACE_INVADERS
    ACTION_SPACE = GAMES.get_action_space(GAME)
    DISPLAY = True

    @staticmethod
    def add_attr(name, value):
        """ Statically adds a parameter as an attribute
        to class Parameters. All new Parameters atributes
        are in capital letters.

        :param name: str
            Name of the new attribute
        :param value: object
            Value of the corresponding hyper-parameter
        """
        name = name.upper()
        setattr(Parameters, name, value)

    @staticmethod
    def load(filepath):
        """ Statically loads the hyper-parameters from a json file

        :param filepath: str
            Path to the json parameter file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
            for key in data.keys():
                if type(data[key]) == dict:
                    Parameters.add_attr(key, data[key]["value"])
