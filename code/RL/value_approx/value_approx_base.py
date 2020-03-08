"""
   Interface for value function approximation in RL.
   It is designed both for model-free prediciton and model-free control.
"""

from abc import ABC, abstractmethod
import numpy as np

class Value_Function_Approximation(ABC) :
    """ Abstract Class for Value Function approximation in RL.
    
        @param  episodes_data   list    List of episodes that are dict of type ((state, action, time_step) : (next_state, reward)).
        @param  gamma   float   Discount factor.
    """
    
    def __init__(self, episodes_data : dict, gamma : float) :
    
    self.episodes_data = episodes_data
    self.gamma = gamma
    # list of all visited states
    self.states_list = self.__get_states_list()
    self.states2idx = self.__get_states_idx()
    
    def __get_states_idx(self) :
        """
            Helper method to match all states with an index.
        """
        dict = {state:None for state in self.states_list}
        for idx, state in enumerate(self.states_list) :
            dict[state] = idx
        return dict

    def get_state_feature(self,state)
        """ Method to get the table lookup features given a state.
        """
        for idx,s in enumerate(self.states_list) :
            if s==state :
                state_idx = idx
        feature_vector = np.array([0]*len(self.states_list),dtype =float)
        feature_vector[i=state_idx] = 1
        
        return feature_vector
        
    def get_state_value_function_approx(self,w)
        """ Method to get the estimate of the state value function given w.
        """
        v = np.array([0]*len(self.states_list), dtype = float)
        # We iterate over each state and use the estimation formula.
        for idx, s in enumerate(self.states_list) :
            v[idx] = np.dot(self.get_state_feature(s), w)
        return v
        
    """ NOTE : All the methods below are also used in model-free prediction and control
        classes.
    """
        
    def get_episode_horizon(self, episode) :
        """ Method to retrieve the lenght of an episode.
        """
        return len(episode)
    
    def __get_states_list(self) :
        """ Method to get the list of visited states.
        """
        ep_num = len(self.episodes_data)
        l = []
        for i in range(ep_num) :
            episode_states = []
            for k in self.episodes_data[i].keys() :
                episode_states.append(k[0])
            l += episode_states
        return list(set(l))

    def get_reward(self, episode) :
        """ Method to get the reward vector for a given episode.
        """
        return [v[1] for __,v in episode.items()]
        
    def get_return(self, episode) :
        """ Returns the discounted returns over all the duration of the
            episode as an array.
            It is equivalent to G_t_inftny in n-steps TD learning.
        """
        horizon = self.get_episode_horizon(episode)
        returns = [0]*horizon
        for i in range(horizon) :
            for j,v in enumerate(episode.values()) :
                if j>=i :
                    returns[i] += v[1]*(self.gamma)**(j-i)
        return returns
        
    def get_return_n(self, episode, step, value_function) :
        """ Used for TD(lambda). It returns G_t_n where n is the step in
            n-steps TD learning.
            
            @param  episode
            @param  step
            @param  value_function
        """
        # Initalization.
        R = self.get_reward(episode)
        horizon = self.get_episode_horizon(episode)
        G_n = [0]*horizon
        
        # Compute the values.
        for t in range(horizon) :
            for lag in range(t,horizon) :
                if lag <= t+step-1 :
                    G_n[t] += R[lag] * (self.gamma)**(lag-t)
             
            # We need to find S_t+step in order to have the estimate
            # of the value function in this tep.
            for i,key in enumerate(episode.keys()) :
                if i == step :
                    S_tstep = key[0]
            # We add the estimate of the value function.
            if t + step  <= horizon - 1 :
                G_n[t] += value_function[S_tstep]*(self.gamma)**step
        
        return G_n
            
    def get_lambda_return(self, lambda_par, episode, value_function) :
        """ Returns the lambda-return vector used in forward view of
            TD(lambda).
            
            @param  lambda_par      float
            @param  episode         dict
            @param  value_function  dict
        """
        # Initialization
        horizon = self.get_episode_horizon(episode)
        G_lambda = np.array([0]*horizon, dtype = float)
        # Compute the lambda-returns values.
        for n in range(1,horizon) :
            G_n = np.array(self.get_return_n(episode,n,value_function), dtype = float)
            G_lambda += (1-lambda_par)*(lambda_par**(n-1))*G_n
        return G_lambda
        
    @abstractmethod
    def get_value_function_estimate(self) :
        pass
            
        
    





