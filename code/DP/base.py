""" Interface for Dynamic Programming Algorithms.
This interface is designed for Policy Evaluation, Policy Iteration, Value Iteration.
"""
from abc import ABC, abstractmethod
import numpy as np
import sys
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/src/processes")
from mdp import MDP
from policy import Policy

class DP(ABC) :
    """ Abstract Class for DP algorithms.
    
        @param mdp_data    dict   Complete MDP data with all probability transitions.
        @param policy_data dict   Policy Data with all probability transisitons.
        @param gamma       float  Discount factor.
        @param max_iter    int    Maximum number of iterations.
    """

    def __init__(self, mdp_data : dict, policy_data : dict, gamma : float,
                 max_iter : int) :
    
        self.mdp_data = mdp_data
        self.policy_data = policy_data
        self.gamma = gamma
        self.max_iter = max_iter
        self.MDP = MDP(mdp_data,gamma)
        self.actions_list = self.__get_actions_list()
        # Policy object (cf processes/policy.py)
        self.Policy = Policy(policy_data)
        # Policy data in a matrix of dim (number_of_action, number_of_states)
        self.pi_matrix = self.__get_pi_matrix()
        # Rewards in a matrix of dim (number_of_states, number_of_actions)
        self.reward_matrix = self.__get_reward_matrix()

    
    def __get_states_list(self) :
        return list(set(state for state in self.mdp_data.keys()))
        
    def __get_actions_list(self) :
        """ Method to retrieve all the possible actions for the input MDP.
        """
        res = []
        for k,_ in self.policy_data.items() :
            for action in self.policy_data[k].keys() :
                res.append(action)
        res = list(set(res))
        res.sort()
        return res
        
        
    def __get_pi_matrix(self) :
        # number of states
        n = len(self.MDP.all_states)
        # number of different actions
        m = len(self.actions_list)
        # reward Matrix
        action2idx = dict.fromkeys(self.actions_list, 0)
        for idx,action in enumerate(self.actions_list) :
            action2idx[action] = idx
        pi = np.zeros((n,m))
        for row_idx,s in enumerate(self.MDP.all_states ):
            row = [0]*m
            for a in self.policy_data[s].keys() :
                idx = action2idx[a]
                row[idx] = self.policy_data[s][a]
            pi[row_idx] = row
          
        return pi.transpose()
        
    def __get_reward_matrix(self) :
        # number of states
        n = len(self.MDP.all_states)
        # number of different actions
        m = len(self.actions_list)
        # reward Matrix
        action2idx = dict.fromkeys(self.actions_list, 0)
        for idx,action in enumerate(self.actions_list) :
            action2idx[action] = idx
        R = np.zeros((n,m))
        for row_idx,s in enumerate(self.MDP.all_states ):
            row = [0]*m
            for a in self.MDP.rewards[s].keys() :
                idx = action2idx[a]
                row[idx] = self.MDP.rewards[s][a]
            R[row_idx] = row
          
        return R
        
    def get_trans_matrix_on_action(self, action) :
        """ Method to retrieve the transition matrix if `action` is done for all states.
        """
        # number of states
        n = len(self.MDP.all_states)
        # returned transition matrix
        P = np.zeros((n,n))
        # compute the values of the matrix
        for k,v in self.mdp_data.items() :
            state = k
            for q,s in v.items() :
                if q==action :
                    for next_state in s[0].keys() :
                        P[state-1,next_state-1] = s[0][next_state]
        
        return P
        

    @abstractmethod
    def get_value_function_estimate(self) :
        pass
        
    @abstractmethod
    def get_optimal_value_function_estimate(self) :
        pass





