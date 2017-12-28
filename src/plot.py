# -*- coding: utf-8 -*-
# plot.py : Graphs, figures and manifold learning
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

import os
import numpy as np
import matplotlib.pyplot as plt


def warn_if_empty(func):
    def new_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except IndexError:
            print("[Warning] Could not save figure in %s: empty sequence" % func.__name__)
    return new_func


class Plotter:

    Q_VALUES = list()
    Q_VALUES_CURRENT_BATCH = list()
    SCORES_PER_EPISODE = list()

    @staticmethod
    def add_episode_score(score):
        Plotter.SCORES_PER_EPISODE.append(score)

    @staticmethod
    def add_q_values_at_t(q_values):
        Plotter.Q_VALUES_CURRENT_BATCH.append(np.squeeze(q_values))

    @staticmethod
    def notify_batch():
        if len(Plotter.Q_VALUES_CURRENT_BATCH) > 0:
            avg_q_values = np.nan_to_num(np.asarray(Plotter.Q_VALUES_CURRENT_BATCH)).mean(axis = 0)
            Plotter.Q_VALUES.append(avg_q_values)
            Plotter.Q_VALUES_CURRENT_BATCH = list()

    @staticmethod
    @warn_if_empty
    def plot_e_scores():
        plt.plot(Plotter.SCORES_PER_EPISODE)
        plt.xlabel("Episode")
        plt.ylabel("Average score per episode")

    @staticmethod
    @warn_if_empty
    def plot_q_values():
        avg_q_values = np.asarray(Plotter.Q_VALUES).mean(axis=1)
        plt.plot(avg_q_values)
        plt.xlabel("Number of batches")
        plt.ylabel("Average Q-values")

    @staticmethod
    def save_plots(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        Plotter.plot_e_scores()
        plt.savefig(os.path.join(folder, "e_scores.png"))
        plt.clf(); plt.close()

        Plotter.plot_q_values()
        plt.savefig(os.path.join(folder, "q_values.png"))
        plt.clf(); plt.close()

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
        

class EmbeddingProjector:
    pass # TODO: manifold learning (Robin: PCA <3)
