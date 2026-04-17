import torch.nn as nn
import torch
from model import Model
from RolloutBuffer import RolloutBuffer
from torch.distributions import Categorical, Normal, Beta


"""
This uses PPO or proximal policy optimization
using MLP as actor
"""
class PPOAgent:
    def __init__(self, num_inputs, num_phases, lr_actor, lr_critic, gamma):

        # this set our device cuda means were using gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Runnnin on {self.device}")

        # Hyperparams
        self.gamma = gamma # based on my understanding gamma is how much it cares on the future also called discount factor
        self.K_epochs = 20 # how long does we use the buffer before clearing it
        self.eps_clip = 0.2 # this limits the change of the ai or policy by more than 20%
        self.gae_lambda = 0.95

        self.buffer = RolloutBuffer() # again the buffer

        self.policy = Model(num_inputs, num_phases).to(self.device) # initialize the mlp
        # set the parameter for each layer learning rate
        # so basically adam is the one who change the weihts and biases in the mlp we provide the parameters 
        # learning rate is how big those changes are lr of actor should be careful slow but lr of critic should be fast 
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.hidden1.parameters(), 'lr': lr_actor},
            {'params': self.policy.hidden2.parameters(), 'lr': lr_actor},
            {'params': self.policy.phase_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.time_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # so that we can compare the old and new policy for the loss calculation since we do not allow big changes in the policy
        self.policy_old = Model(num_inputs, num_phases).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss() # calculate the mean squared loss

    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  
        with torch.no_grad():
            phase_probs, time_params, state_value = self.policy_old(state)

            phase_dist = Categorical(phase_probs)
            phase_action = phase_dist.sample()

            time_dist = Beta(time_params[:, 0], time_params[:, 1])  
            time_action = time_dist.sample()

        self.buffer.states.append(state)
        self.buffer.actions_phase.append(phase_action)
        self.buffer.actions_time.append(time_action)

        self.buffer.logprobs_phase.append(phase_dist.log_prob(phase_action))
        self.buffer.logprobs_time.append(time_dist.log_prob(time_action))
        self.buffer.state_values.append(state_value)

        return phase_action.item(), time_action.item()
    
    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions_phase = torch.squeeze(torch.stack(self.buffer.actions_phase, dim=0)).detach().to(self.device)
        old_actions_time = torch.squeeze(torch.stack(self.buffer.actions_time, dim=0)).detach().to(self.device)
        old_logprobs_phase = torch.squeeze(torch.stack(self.buffer.logprobs_phase, dim=0)).detach().to(self.device)
        old_logprobs_time = torch.squeeze(torch.stack(self.buffer.logprobs_time, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals

        advantages_list = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = old_state_values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1.0 - is_terminals[i]) - old_state_values[i]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - is_terminals[i]) * gae
            advantages_list.insert(0, gae)

        advantages = torch.stack(advantages_list).to(self.device)
        returns = advantages + old_state_values # What the Critic SHOULD have guessed

        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)


        for _ in range(self.K_epochs): # we learned k epochs meaning k epochs is our batch size

            new_phase_probs, new_time_params, new_state_values = self.policy(old_states)

            new_phase_dist = Categorical(new_phase_probs)
            new_time_dist = Beta(new_time_params[:, 0], new_time_params[:, 1])

            new_logprobs_phase = new_phase_dist.log_prob(old_actions_phase)
            new_logprobs_time = new_time_dist.log_prob(old_actions_time)

            new_logprobs = new_logprobs_phase + new_logprobs_time
            old_logprobs  = old_logprobs_phase + old_logprobs_time

            # we take the exponential diff of old and new probs
            # this shows how much our model changed
            ratios = torch.exp(new_logprobs - old_logprobs)

            # surrogative objective
            surr1 = ratios * advantages # this is how much the ai wants to incresse or decrease the ratio
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages # we clamp it since remember we dont want the ai to be greedy and let it learn slowly

            entropy = new_phase_dist.entropy() + new_time_dist.entropy()
            loss =  -torch.min(surr1, surr2) + 0.5 * self.MseLoss(new_state_values.squeeze(), returns) - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        device_loc = self.device 
        
        checkpoint = torch.load(checkpoint_path, map_location=device_loc)
        
        self.policy_old.load_state_dict(checkpoint)
        self.policy.load_state_dict(checkpoint)

    def get_deterministic_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            with torch.no_grad():
                phase_probs, time_params, _ = self.policy_old(state)

                chosen_phase = torch.argmax(phase_probs, dim=-1).item()

                alpha = time_params[:, 0]
                beta = time_params[:, 1]
                chosen_time_percentage = (alpha / (alpha + beta)).item()

            return chosen_phase, chosen_time_percentage



    





