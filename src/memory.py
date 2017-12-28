
from parameters import Parameters
import numpy as np

class Memory:

    def __init__(self):

        self.memory_size = Parameters.REPLAY_MEMORY_SIZE
        self.current_memory_index = 0
        self.memory_usage = 0

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.screens = np.empty((self.memory_size, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH), dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        self.minibatch_size = Parameters.MINIBATCH_SIZE

        state_shape = (self.minibatch_size, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH)
        self.state_t = np.empty(state_shape, dtype=np.float16)
        self.state_t_plus_1 = np.empty(state_shape, dtype=np.float16)


    def add(self, screen, action, reward, terminal):

        self.actions[self.current_memory_index] = action
        self.rewards[self.current_memory_index] = reward
        self.screens[self.current_memory_index, ...] = screen
        self.terminals[self.current_memory_index] = terminal

        self.memory_usage = max(self.memory_usage, self.current_memory_index + 1)
        self.current_memory_index = (self.current_memory_index + 1) % self.memory_size

    
    def get_state(self, state_index):

        state = None

        if(not self.memory_usage):
            print("Memory is empty")
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

            memory = np.random.randint(Parameters.AGENT_HISTORY_LENGTH, self.memory_usage)
            
            if(not self.includes_terminal(memory) and not self.includes_current_memory_index(memory)):
                self.state_t[len(selected_memories), ...] = self.get_state(memory-1)
                self.state_t_plus_1[len(selected_memories), ...] = self.get_state(memory)
                selected_memories.append(memory)
        
        # for compatibility issues
        selected_memories = np.array(selected_memories)
        
        return(self.state_t, self.actions[selected_memories], self.rewards[selected_memories], self.state_t_plus_1, self.terminals[selected_memories])
        

    def includes_terminal(self, index):
        return(np.any(self.terminals[(index - Parameters.AGENT_HISTORY_LENGTH):index]))


    def includes_current_memory_index(self, index):
        return((index >= self.current_memory_index) and (index - Parameters.AGENT_HISTORY_LENGTH < self.current_memory_index))

    
    def get_usage(self):
        return(self.memory_usage)
