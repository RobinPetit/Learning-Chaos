# -*- coding: utf-8 -*-
# parameters.py : Load parameters from json config file
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from games import Games

import json


class Parameters:

    # environment settings
    GAMES = Games()
    GAME = "SpaceInvaders-v0"
    DISPLAY = False

    USE_DDDQN = True
    USE_PRIORITIZATION = False

    LOADED_FILE = None

    TO_UPDATE = None
    CURRENT_STEP = 0

    @staticmethod
    def add_attr(name, value):
        """ Statically adds a parameter as an attribute
        to class Parameters. All new Parameters attributes
        are in capital letters.

        :param name: str
            Name of the new attribute
        :param value: object
            Value of the corresponding hyper-parameter
        """
        name = name.upper()
        setattr(Parameters, name, value)

    @staticmethod
    def get_attr(name):
        return getattr(Parameters, name.upper())

    @staticmethod
    def load(filepath):
        """ Statically loads the hyper-parameters from a json file

        :param filepath: str
            Path to the json parameter file
        """
        Parameters.LOADED_FILE = filepath
        Parameters.TO_UPDATE = list()
        with open(filepath, "r") as f:
            data = json.load(f)
            for key in sorted(data.keys()):
                if isinstance(data[key], dict):
                    if key in ("SESSION_SAVE_FILENAME", "SESSION_SAVE_DIRECTORY"):
                        # make session path specific to the game being played
                        data[key]["value"] = data[key]["value"] + \
                            Parameters.GAME
                    elif key == "GAME":
                        data[key]["value"] = eval(
                            'Parameters.GAMES.' + data[key]["value"])
                    Parameters.add_attr(key, data[key]["value"])
                    if "update" in data[key] and data[key]["update"]:
                        Parameters.TO_UPDATE.append(key)

    @staticmethod
    def update():
        if Parameters.LOADED_FILE is None:
            print('[Warning] Trying to save parameters but none have been loaded.')
            return
        with open(Parameters.LOADED_FILE, "r") as f:
            data = json.load(f)
            for key in data:
                if not isinstance(data[key], dict) or key not in Parameters.TO_UPDATE:
                    continue
                if data[key]['value'] != Parameters.get_attr(key):
                    data[key]["value"] = Parameters.get_attr(key)
        with open(Parameters.LOADED_FILE, "w") as f:
            pretty_str = json.dumps(data, indent=4, sort_keys=True)
            f.write(pretty_str)
