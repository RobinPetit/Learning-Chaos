
from games import Games

class Parameters:

    # environment settings
    GAMES = Games()
    GAME = games.SPACE_INVADERS
    ACTION_SPACE = games.get_action_space(game)
    DISPLAY = True

    @staticmethod
    def add_attr(name, value):
        """ Statically add a parameter as an attribute

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