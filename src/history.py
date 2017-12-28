# -*- coding: utf-8 -*-
# history.py  : Experience replay
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

"""
[Article] The function w from algorithm ϕ described below applies this preprocess-
ing to the m most recent frames and stacks them to produce the input to the
Q-function, in which m = 4, although the algorithm is robust to different values of
m (for example, 3 or 5).
"""

from parameters import Parameters

import numpy as np


class FramesHistory:

    """ The frame history will be used as an input to the convolutional neural network """

    def __init__(self):
            
        """ /!\ Always in Float32 otherwise cv2.resize shits itself /!\ """
        history_shape = (Parameters.AGENT_HISTORY_LENGTH, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH)
        self.frames_history = np.zeros(history_shape, dtype=np.float32)


    def add_frame(self, frame):

        """
        Add a frame to the frame history
        Sadly, custom slicing is faster than np.roll()
        Maybe a faster version can be achieved by using np.lib.stride_tricks.as_strided 
        """
        self.frames_history[:-1] = self.frames_history[1:]
        self.frames_history[-1] = frame


    def reset(self):
        self.frames_history *= 0

    
    def get(self):
        # Return array of shape (IMAGE_HEIGHT, IMAGE_WIDTH, AGENT_HISTORY_LENGTH)
        history = np.copy(self.frames_history)
        history = np.swapaxes(history, 0, 1)
        history = np.swapaxes(history, 1, 2)
        return history



        
        
