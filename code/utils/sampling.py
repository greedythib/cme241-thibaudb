""" Helper functions for manipulating MDP and RL algorithms.  
"""
import sys
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/code/processes")
from mdp import MDP
import numpy as np
from policy import Policy
import random

def MDP_sampling(mdp_data, policy_data, ep_num, max_ep_length = 500 ,gamma = 0.5) :
    """
    Creates a random sequence of episodes.
    
    @param    mdp_data     dict  {states : {actions : prob_next_state}}.
    @param    policy_data  dict    {state : {action : prob_action}}.
    @param    ep_num       int   Number of episodes in the simulation data.
    @param    seq_size     int   Number of sequences generated within one episode.
    
    Returns list of {(state, action, time_step) : (next_state,reward)}  :  An uncontrolled sequence.
    """
    # Gets the MDP from the input data
    mdp = MDP(mdp_data, gamma)
    # Simulate a random sequence.
    simulation_data = []
    for ep in range(ep_num) :
        # Choose a random size of episode
        episode_size = np.random.randint(1,max_ep_length)
        # Create a random episode sequence
        uncontrolled_sequence = dict()
        ## 1. Choses a random state.
        state = random.randint(1,len(mdp.all_states))
        for k in range(episode_size) :
            ## 2. Choses a random action by using the policy distribution.
            actions_list = list(policy_data[state].keys())
            distribution = list(policy_data[state].values())
            action = np.random.choice(actions_list,1,p=distribution)[0]
            ## 3. Choses the next state based on the MDP distribution.
            reward = mdp_data[state][action][1]
            next_states_list = list(mdp_data[state][action][0].keys())
            next_state_distribution = list(mdp_data[state][action][0].values())
            next_state = np.random.choice(next_states_list,1,p=next_state_distribution)[0]
            uncontrolled_sequence[(state,action,k)] = (next_state, reward)
            state = next_state 
        
        simulation_data.append(uncontrolled_sequence)
        
    return simulation_data
    
    
if __name__ == '__main__':

 policy_data = {

         1: {'a': 0.4, 'b': 0.6},
         2: {'a': 0.7, 'c': 0.3},
         3: {'a' : 0.5, 'b': 0.5}
 }

 data = {
     1: {
         'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
         'b': ({2: 0.3, 3: 0.7}, 2.8),
         'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
     },
     2: {
         'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 5.0),
         'c': ({1: 0.2, 2: 0.6, 3: 0.2}, -7.2)
     },
     3: {
         'a': ({1:0.5, 3: 0.5}, 1.0),
         'b': ({2: 0.5, 3:0.5}, 10)
     }
 }
 
 print("SANITY CHECKS")
 print("-"*80)
 print(MDP_sampling(data, policy_data, ep_num =5))
 

