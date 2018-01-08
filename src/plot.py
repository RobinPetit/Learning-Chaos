# -*- coding: utf-8 -*-
# plot.py : Graphs, figures and manifold learning
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import cv2

import gym
from parameters import Parameters
from sklearn.manifold import TSNE

# Set to True if the Q values plot should display the individual
# Q values in addition to the mean
PLOT_INDIVIDUAL_Q_VALUES = False


def warn_if_empty(func):
    def new_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except IndexError:
            print(
                "[Warning] Could not save figure in %s: empty sequence" %
                func.__name__)
    return new_func


def moving_average(sequence, n=1000):
    ma = np.cumsum(sequence, axis=0)
    ma[n:] = ma[n:] - ma[:-n]
    return ma[n - 1:] / n


class Plotter:

    Q_VALUES = list()
    Q_VALUES_CURRENT_BATCH = list()
    SCORES_PER_EPISODE = list()

    @staticmethod
    def add_episode_score(score):
        Plotter.SCORES_PER_EPISODE.append(score)

    @staticmethod
    def add_q_values_at_t(q_values):
        Plotter.Q_VALUES.append(np.squeeze(q_values))

    @staticmethod
    def notify_batch():
        if len(Plotter.Q_VALUES_CURRENT_BATCH) > 0:
            avg_q_values = np.nan_to_num(
                np.asarray(Plotter.Q_VALUES_CURRENT_BATCH)).mean(axis=0)
            Plotter.Q_VALUES.append(avg_q_values)
            Plotter.Q_VALUES_CURRENT_BATCH = list()

    @staticmethod
    @warn_if_empty
    def plot_e_scores():
        smoothed = moving_average(Plotter.SCORES_PER_EPISODE)
        plt.plot(smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Average score per episode")

    @staticmethod
    @warn_if_empty
    def plot_q_values():
        avg_q_values = np.asarray(Plotter.Q_VALUES)
        smoothed = moving_average(avg_q_values, n=5000).T
        if PLOT_INDIVIDUAL_Q_VALUES:
            action_names = gym.envs.make(Parameters.GAME).unwrapped.get_action_meanings()
            for idx, smoothed_q_value in enumerate(smoothed):
                plt.plot(smoothed_q_value, label=action_names[idx])
        plt.plot(np.squeeze(smoothed.mean(axis=0)), label='MEAN')
        plt.legend(loc='best')
        plt.xlabel("Number of batches")
        plt.ylabel("Average Q-values")

    @staticmethod
    def save_plots(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        Plotter.plot_e_scores()
        plt.savefig(os.path.join(folder, "e_scores.png"))
        plt.clf()
        plt.close()

        Plotter.plot_q_values()
        plt.savefig(os.path.join(folder, "q_values.png"))
        plt.clf()
        plt.close()

    @staticmethod
    def load(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            e_scores_path = os.path.join(folder, "e_scores.npy")
            q_values_path = os.path.join(folder, "q_values.npy")
            if os.path.isfile(q_values_path):
                Plotter.Q_VALUES = list(np.load(q_values_path))
            if os.path.isfile(e_scores_path):
                Plotter.SCORES_PER_EPISODE = list(np.load(e_scores_path))

    @staticmethod
    def save(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        e_scores_path = os.path.join(folder, "e_scores.npy")
        q_values_path = os.path.join(folder, "q_values.npy")
        np.save(e_scores_path, np.asarray(Plotter.SCORES_PER_EPISODE))
        np.save(q_values_path, np.asarray(Plotter.Q_VALUES))

    @staticmethod
    def reset(folder):
        for name in os.listdir(folder):
            filepath = os.path.join(folder, name)
            if filepath.endswith(".png") or filepath.endswith(".npy"):
                os.remove(filepath)

    @staticmethod
    def plot_conv_layers(agent):

        folder = "./out/layers_plots/" + str(Parameters.GAME) + "/" + str(agent.step) + "/"

        filters = {"conv1": 32, "conv2": 64, "conv3": 64}
        filters_scale = {"conv1": 10, "conv2": 20, "conv3": 30}

        if not os.path.exists(folder):
            os.makedirs(folder)

        # get a random state to feed the CNN
        state_t, _, _, _, _, _, _ = agent.memory.bring_back_memories()

        # we take image 0
        img = 0

        for i in range(Parameters.AGENT_HISTORY_LENGTH):
            plt.imsave(folder + "dqn_input_" + str(i) + ".png", state_t[img,:,:,i], cmap='gray')

        # extract the output of each CNN layer for that state
        for layer in ["conv1", "conv2", "conv3"]:

            layer_folder = folder + layer + "/"
            if not os.path.exists(layer_folder):
                os.makedirs(layer_folder)

            layer_output = agent.tf_session.run(agent.dqn.layers[layer], {agent.dqn_input: state_t})

            scale = filters_scale[layer]
            for filter in range(filters[layer]):
                plt.imsave(layer_folder + layer + "_filter_" + str(filter+1) + ".png",
                    cv2.resize(layer_output[img,:,:,filter], (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cmap='gray')



class EmbeddingProjector(TSNE):
    def __init__(self, *args, **kwargs):
        TSNE.__init__(self, *args, **kwargs)

    def save_plot(self, states, hidden_repr, v_values, folder):
        projected = self.fit_transform(hidden_repr)
        print('projected\n\tCreating display')

        # create different axes of different size
        ax_tsne = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
        ax_min  = plt.subplot2grid((2, 3), (0, 2))
        ax_max  = plt.subplot2grid((2, 3), (1, 2))
        # get related figure to add the colorbar of t-SNE
        fig = plt.figure(plt.get_fignums()[-1])
        # create an axis dedicated to the colorbar
        cbaxis = fig.add_axes([0.01, 0.1, 0.02, 0.8])
        MIN_COLOR = 'b'
        MAX_COLOR = 'r'
        edgecolors = ['none'] * len(projected)
        edgecolors[v_values.argmin()] = MIN_COLOR
        edgecolors[v_values.argmax()] = MAX_COLOR
        points = ax_tsne.scatter(projected[:, 0], projected[:, 1], c=v_values, cmap='summer', edgecolors=edgecolors, lw=1)
        plt.colorbar(points, cax=cbaxis)
        ax_tsne.axis('off')

        ax_min.imshow(states[v_values.argmin()], cmap='gray', interpolation='none')
        set_axis_color(ax_min, MIN_COLOR)
        ax_min.set_xticks([])
        ax_min.set_yticks([])

        ax_max.imshow(states[v_values.argmax()], cmap='gray', interpolation='none')
        set_axis_color(ax_max, MAX_COLOR)
        ax_max.set_xticks([])
        ax_max.set_yticks([])

        # add edges connecting points on t-SNE and boxes on the right
        ax_tsne.add_artist(ConnectionPatch(xyA=projected[v_values.argmin()], xyB=(1,10), coordsA="data", coordsB="data",
            axesA=ax_tsne, axesB=ax_min, color=MIN_COLOR, lw=1, ls='dashed'))
        ax_tsne.add_artist(ConnectionPatch(xyA=projected[v_values.argmax()], xyB=(1,10), coordsA="data", coordsB="data",
            axesA=ax_tsne, axesB=ax_max, color=MAX_COLOR, lw=1, ls='dashed'))

        plt.suptitle("t-distributed Stochastic Neighbor Embedding")
        plt.savefig(os.path.join(folder, "tsne.png"), dpi=144)

def set_axis_color(ax, col):
    for key in ('bottom', 'top', 'right', 'left'):
        ax.spines[key].set_color(col)
