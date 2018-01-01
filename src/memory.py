# -*- coding: utf-8 -*-
# memory.py : Agent's short term and long term memory
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from parameters import Parameters
import numpy as np

import os
from shutil import copyfile, move
import shelve

DEFAULT_MEMMAP_PATH = "mem.dat"
DEFAULT_SAVE_PATH = "memory-" + Parameters.GAME + '.shelf'

class ShortTermMemory:

    def __init__(self, mmap, memory):
        """
        :param mmap: np.memmap
            Memory map that holds all the experience samples.
            This map represents the long-term memory.
        """
        self.parent_memory = memory
        self.long_term_mem = mmap
        buffer_shape = list(mmap.shape)
        buffer_shape[0] = Parameters.SHORT_TERM_MEMORY_SIZE
        self.buffer = np.zeros(shape=buffer_shape)


    def sample_random(self):
        random_indices = np.random.choice(self.parent_memory.memory_usage, self.buffer.shape[0])
        self.buffer = self.long_term_mem[random_indices, :]


class Memory:

    def __init__(self, destination=DEFAULT_MEMMAP_PATH, load=True):
        """
        :param destination: str
            Path to the file where the long-term experience must be stored
            Note: Files cannot be larger than 2 Gb in 32-bit architectures
        :param load: bool
            True by default. Should be False only when called by child class
            that call load_memory on its own!
        """

        self.memory_filepath = destination

        self.memory_size = Parameters.LONG_TERM_MEMORY_SIZE

        screens_shape = (self.memory_size, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH)
        self.screens = np.memmap(self.memory_filepath, mode="w+", shape=screens_shape, dtype=np.float16)
        self.minibatch_size = Parameters.MINIBATCH_SIZE
        self.state_shape = (self.minibatch_size, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH)
        if load:
            self.load_memory()


    def save_memory(self, path=DEFAULT_SAVE_PATH):
        with shelve.open(path) as shelf:
            shelf["idx"] = self.current_memory_index
            shelf["mem usage"] = self.memory_usage
            shelf["actions"] = self.actions
            shelf["rewards"] = self.rewards
            shelf["terminals"] = self.terminals
            shelf["state t"] = self.state_t
            shelf["state t+1"] = self.state_t_plus_1
        self.screens.flush()


    def load_memory(self, path=DEFAULT_SAVE_PATH):
        try:
            with shelve.open(path) as shelf:
                self.current_memory_index = shelf["idx"]
                self.memory_usage = shelf["mem usage"]
                self.actions = shelf["actions"]
                self.rewards = shelf["rewards"]
                self.terminals = shelf["terminals"]
                self.state_t = shelf["state t"]
                self.state_t_plus_1 = shelf["state t+1"]
            return True
        except KeyError:
            self.current_memory_index = 0
            self.memory_usage = 0
            self.actions = np.empty(self.memory_size, dtype=np.uint8)
            self.rewards = np.empty(self.memory_size, dtype=np.integer)
            self.terminals = np.empty(self.memory_size, dtype=np.bool)
            self.state_t = np.empty(self.state_shape, dtype=np.float16)
            self.state_t_plus_1 = np.empty(self.state_shape, dtype=np.float16)
            print('Created new memory')
            try:
                os.remove(self.memory_filepath)
                os.remove(self.memory_filepath+'.bak')
            except FileNotFoundError:
                pass  # remove files if they exist so if they don't exist, just ignore
            return False


    def add(self, screen, action, reward, terminal):
        # q_estimates are not used by the Memory, but by the PrioritizedMemory

        self.actions[self.current_memory_index] = action
        self.rewards[self.current_memory_index] = reward
        self.screens[self.current_memory_index, ...] = screen
        self.terminals[self.current_memory_index] = terminal

        self.memory_usage = max(self.memory_usage, self.current_memory_index + 1)
        self.current_memory_index = (self.current_memory_index + 1) % self.memory_size


    def get_state(self, state_index):

        state = None

        if(not self.memory_usage):
            print("[Warning] Memory is empty")
        else:

            state_index = state_index % self.memory_usage

            if(state_index >= Parameters.AGENT_HISTORY_LENGTH - 1):
                state = self.screens[(state_index - Parameters.AGENT_HISTORY_LENGTH) + 1 : state_index + 1, ...]
            else:
                # negative indices don't work well with slices in numpy..
                state = self.screens[np.array([(state_index-i) % self.memory_usage for i in reversed(range(Parameters.AGENT_HISTORY_LENGTH))]), ...]

        state = np.swapaxes(state, 0, 1)
        state = np.swapaxes(state, 1, 2)

        return(state)

    def sample_memory(self, nb_samples=1):
        """
        :param nb_samples: int
            Number of integers to return
        """
        return Parameters.AGENT_HISTORY_LENGTH + np.random.choice(self.memory_usage-Parameters.AGENT_HISTORY_LENGTH, nb_samples, replace=False)

    def bring_back_memories(self):
        """
        [Article] our algorithm only stores the last N experience tuples in the replay
        memory, and samples uniformly at random from D when performing updates. This
        approach is in some respects limited because the memory buffer does not differ-
        entiate important transitions and always overwrites with recent transitions owing
        to the finite memory size N. Similarly, the uniform sampling gives equal impor-
        tance to all transitions in the replay memory. A more sophisticated sampling strat-
        egy might emphasize transitions from which we can learn the most, similar to
        prioritized sweeping

        This method selects and returns [minibatch_size] memories randomly from the memory
        Memories that contain either a terminal or the self.current_memory_index are not selected
        """
        assert(self.memory_usage > Parameters.AGENT_HISTORY_LENGTH)

        selected_memories = []

        while(len(selected_memories) < self.minibatch_size):

            memories = self.sample_memory(self.minibatch_size - len(selected_memories))

            for memory in memories:
                if(not self.includes_terminal(memory) and not self.includes_current_memory_index(memory)):
                    self.state_t[len(selected_memories), ...] = self.get_state(memory-1)
                    self.state_t_plus_1[len(selected_memories), ...] = self.get_state(memory)
                    selected_memories.append(memory)

        # for compatibility issues
        selected_memories = np.array(selected_memories)

        return(
            self.state_t,
            self.actions[selected_memories],
            self.rewards[selected_memories],
            self.state_t_plus_1,
            self.terminals[selected_memories],
            self.get_importance_sampling_weights(selected_memories),
            selected_memories
        )

    def update(self, memory_indices, q_estimates, losses, completion):
        pass # Regular memory is uniformly random

    def get_importance_sampling_weights(self, memory_indices):
        # No importance sampling for regular memory
        return np.ones(len(memory_indices), dtype=np.float32)

    def includes_terminal(self, index):
        return(np.any(self.terminals[(index - Parameters.AGENT_HISTORY_LENGTH):index]))


    def includes_current_memory_index(self, index):
        return((index >= self.current_memory_index) and (index - Parameters.AGENT_HISTORY_LENGTH < self.current_memory_index))


    def get_usage(self):
        return(self.memory_usage)


    @staticmethod
    def reset(path=DEFAULT_MEMMAP_PATH):
        if os.path.exists(path):
            os.remove(path)


class PrioritizedMemory(Memory):
    """
    References
    ----------
    https://arxiv.org/pdf/1511.05952.pdf
    Prioritized experience replay
    Schaul et al.
    """
    def __init__(self, alpha=1.0, beta_0=2.0, epsilon=0.0, p_0=1.0, destination="mem.dat"):
        """
        :param alpha: float
            Degree of prioritization
            In the case of uniform experience replay, alpha = 0
        :param beta_0: float
            Initial value of beta, which is the exponent of importance-sampling weights
        :param epsilon: float
            Minimum priority (of any experience sample).
            If epsilon > 0, this ensures that every experience sample
            has a non-zero probability of being selected.
        :param p_0: float
            Initial value for all priorities
        :param destination: str
            Path to the file where the long-term experience must be stored
            Note: Files cannot be larger than 2 Gb in 32-bit architectures
        """
        Memory.__init__(self, destination=destination, load=False)
        self.alpha = alpha
        self.beta = self.beta_0 = beta_0
        self.epsilon = epsilon
        self.p_0 = p_0
        self.load_memory()


    def save_memory(self, path=DEFAULT_SAVE_PATH):
        Memory.save_memory(self, path)
        with shelve.open(path) as shelf:
            shelf["priorities"] = self.priorities
            shelf["sampling"] = self.sampling_probs
            shelf["weights"] = self.i_s_weights


    def load_memory(self, path=DEFAULT_SAVE_PATH):
        ret = Memory.load_memory(self, path)
        if not ret:
            self.priorities = np.full(self.memory_size, fill_value=self.p_0, dtype=np.float32)
            self.sampling_probs = np.ones(self.memory_size, dtype=np.float64)
            self.i_s_weights = np.ones(self.memory_size, dtype=np.float64)
        else:
            with shelve.open(path) as shelf:
                self.priorities = shelf["priorities"]
                self.sampling_probs = shelf["sampling"]
                self.i_s_weights = shelf["weights"]
        return ret


    def sample_memory(self, nb_samples=1):
        self.update_probs_and_weights()
        probs = self.sampling_probs[:self.memory_usage]
        return np.random.choice(self.memory_usage, size=nb_samples, p=probs)

    def update_probs_and_weights(self):
        probs = self.sampling_probs[:self.memory_usage]
        self.sampling_probs[:self.memory_usage] = probs = probs / probs.sum()
        max_w_i = self.i_s_weights[:self.memory_usage]
        self.i_s_weights[:self.memory_usage] = (self.memory_usage * probs) ** (-self.beta) / max_w_i

    def update(self, memory_indices, q_estimates, losses, completion):
        """
        :param memory_indices: np.ndarray[ndim=1]
            Indices of batch samples in memory.
            The array length must be equal to the batch size.
        :param q_estimates: np.ndarray[ndim=2]
            Array of shape (batch_size, n_actions) containing the estimated
            Q-values for each batch sample.
        :param losses: float
            Evaluated loss function for each sample of the current batch
        :param completion: float
            Number of performed learning steps divided by the maximum number of steps
        """
        self.priorities[memory_indices] = np.abs(losses) + self.epsilon
        self.sampling_probs[:self.memory_usage] = self.priorities[:self.memory_usage] ** self.alpha
        self.beta = self.beta_0 - (completion * float(self.beta_0 - 1.0))

    def get_importance_sampling_weights(self, memory_indices):
        return self.i_s_weights[memory_indices]

