
"""
    Basically this serves somewhat a memory of the agent, it take notes what action it  saw from the states iput what it did and what rewards did it get
    but will not be used for the future training it wrote down all of this at the moment, when its time to learn it read this update the model then throw it
    so basically a buffer instead of a memory
"""

class RolloutBuffer:
    def __init__(self):
        self.states = [] # what are the current inputs
        
        self.actions_phase = [] # what did phase it choose
        self.actions_time = [] # for what time 0 - 1 / basically 0% to 100%
        
        self.logprobs_phase = [] # how confident the model when choosing the phase
        self.logprobs_time = [] # how confident the model when choosing the time
        
        self.rewards = [] # reward it get after those
        self.state_values = [] # this is the critic guess for the score of that action
        self.is_terminals = [] # did the simulation end / episode end just a true false flag so no future reward to be expected
    
    def clear(self):
        # as we said we dont use this for future training so we clear everything after the update of the model
        del self.states[:]
        del self.actions_phase[:]
        del self.actions_time[:]
        del self.logprobs_phase[:]
        del self.logprobs_time[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]