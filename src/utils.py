# -*- coding: utf-8 -*-
# utils.py: Image preprocessing
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from parameters import Parameters

import numpy as np
import cv2


def y_channel(image):
    """
    Y-channel = luminance = greyscale = (0.299*R + 0.587*G + 0.114*B)
    orb.essex.ac.uk/ce/ce316/opencv-intro.pdf
    """
    return(np.dot(image[..., :3], [0.299, 0.587, 0.114]))


def screen_resize(image):
    """
    Resize the image with the width and height of the parameters (84x84 in the article)
    Using cv2 and INTER_AREA for speed
    Original size: 210x160x3
    """
    return(cv2.resize(image, (Parameters.IMAGE_WIDTH, Parameters.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))


def remove_flickering(previous_image, image):
    """
    [Article] First, to encode a single frame we take the maximum value for each pixel colour
    value over the frame being encoded and the previous frame. This was necessary to
    remove flickering that is present in games where some objects appear only in even
    frames while other objects appear only in odd frames, an artefact caused by the
    limited number of sprites Atari 2600 can display at once.
    """
    return(np.asarray([previous_image, image]).max(axis=0))

def preprocess_img(previous_image, image):
    #return np.asarray(np.round(screen_resize(y_channel(remove_flickering(previous_image, image)))), dtype=np.uint8)  # uncomment to remove flickering
    return np.asarray(np.round(screen_resize(y_channel(image))), dtype=np.uint8)  # uncomment to leave flickering

def reward_clipper(reward):
    """
    [Article] As the scale of scores varies greatly from game to game, we clipped
    all positive rewards at 1 and all negative rewards at 21, leaving 0 rewards unchanged.
    Clipping the rewards in this manner limits the scale of the error derivatives and
    makes it easier to use the same learning rate across multiple games.
    At the same time, it could affect the performance of our agent since it cannot
    differentiate between rewards of different magnitude.
    """
    return(max(Parameters.NEGATIVE_REWARD, min(Parameters.POSITIVE_REWARD, reward)))
