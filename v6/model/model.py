import torch
import torch.nn as nn
import torch.nn.functional as F

# This is an MLP implementation which output two outputs
# They called it multi head neural network
class Model(nn.Module):
    """
        So it takes number of inputs since it will turn into the
        input layer, the number of phases will be the number of
        neurons in the output layer for the phase
    """
    def __init__(self, num_inputs, num_phases):
        super().__init__() # inherits the class from nn.module


        # this is our hidden layer
        # 64 is a great number since we have 23 inputs having 16 is too low 
        # having 128 is too many and would lead to it just memorizations of all possibilities
        # 64 is a sweet spot, and the reason we use a block hidden layer
        # meaning both 64 instead of the cascading is because we have two output branches and we 
        # dont want to bottleneck or like compress the information so this works fine
        self.hidden1 = nn.Linear(num_inputs, 128)
        self.hidden2 = nn.Linear(128, 128)

        # This will be the branch of the phase neurons
        # it turn 64 neurons into the number of phases
        self.phase_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_phases)
        )      


        # This will be the branch of the time neurons
        # so it will just output one neuron from 0 to 1
        # since were use action scaling to get that
        # dynamic min max time behaviour instead of training in
        # seconds
        self.time_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
        )


        self.critic = nn.Linear(128, 1)

    """
     we feed the data
    """
    def  forward(self, state):
        # so we take our input state
        # car count, waiting time ev present for lanes pedestrian
        # that is our input layer, we pass it now to the hidden layers
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        # now we need to have two output the phase and time so we need to both use them
        #  we take the hiddenlayer2 and feed it to the phase probability
        # so in context softmax turn the sum of the output layer in percentage and would be sum to 100%
        #  so the neuron that has the highest percentage would be the best action in that state
        phase_prob = F.softmax(self.phase_head(x), dim=-1)

        # next is the time percentage we use this so that the seconds can be dynamic depending on
        # the set min or max time same as the phase probs but this time we use sigmoid to make whatever the output
        #  to be between 0 and 1 or 0% to 100% basically
        time_params = F.softplus(self.time_head(x)) + 1.0


        state_value = self.critic(x) # this is what the critic score like it guesses from the hidden layer
        
        return phase_prob, time_params, state_value
    

