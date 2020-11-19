import random
import torch
import numpy as np


class MainMemory:
    def __init__(self, batch_size):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.is_terminals = []
        self.returns = []
        self.batch_size = batch_size

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        self.values = []
        self.advantages = []
        del self.is_terminals[:]
        del self.returns[:]

    def merge_memories(self, other_memory):
        # each process collects its own memory independantly on a new env. Merge all in this main memory
        states, actions, log_prob, rewards, is_terminals = other_memory.get_full_memory()
        self.states += states
        self.actions += actions
        self.logprobs += log_prob
        self.rewards += rewards
        self.is_terminals += is_terminals

    def memory_size(self):
        return len(self.states)

    def critic_values(self, critic, device):
        stacked_states = torch.stack([torch.tensor(state) for state in self.states]).to(device)

        self.values = critic(stacked_states).detach().cpu().numpy()
        self.values = np.append(self.values, [0])

    def get_batch(self):
        indices = random.sample(range(len(self.states)), self.batch_size)

        batch_states = torch.stack([torch.tensor(self.states[i]) for i in indices])
        batch_actions = torch.stack([torch.tensor(self.actions[i]) for i in indices])
        batch_logprobs = torch.stack([torch.tensor(self.logprobs[i]) for i in indices])
        batch_advantages = torch.stack([torch.tensor(self.advantages[i]) for i in indices])
        # batch_returns = torch.stack([torch.tensor(self.returns[i]) for i in indices])

        return batch_states, batch_actions, batch_logprobs, batch_advantages

    def get_value_batch(self):
        indices = random.sample(range(len(self.states)), self.batch_size)

        batch_states = torch.stack([torch.tensor(self.states[i]) for i in indices])
        batch_returns = torch.stack([torch.tensor(self.returns[i]) for i in indices])

        return batch_states, batch_returns


class ProcessMemory:
    def __init__(self, id):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

        self.id = id

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def memory_size(self):
        print("id", self.id)
        return len(self.states)

    def get_full_memory(self):
        return self.states, self.actions, self.logprobs, self.rewards, self.is_terminals
