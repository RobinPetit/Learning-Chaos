
import gym

class Games:

    def __init__(self):   

        # Cool (easy) Atari environments available
        self.ASTEROIDS = "Asteroids-v0"
        self.PONG = "Pong-v0"
        self.SPACE_INVADERS = "SpaceInvaders-v0"
        self.TENNIS = "Tennis-v0"

        self.available_games_list = [self.ASTEROIDS, self.PONG, self.SPACE_INVADERS, self.TENNIS]
        self.action_space = {}
        self.define_action_spaces()


    def define_action_spaces(self):
        
        for game in self.available_games_list:
            dummy_env = gym.envs.make(game)
            self.action_space[game] = dummy_env.action_space.shape[0]

    def get_action_space(self, game):
        return(self.action_space[game])

        