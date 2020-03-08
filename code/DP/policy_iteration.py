""" We want to evaluate a MDP given a policy
"""
import numpy as np
from base import DP

class Policy_Evaluation(DP) :
    """ Derived class of DP class to implemente Policy_Evaluation algorithm.
    """
    
    def get_value_function_estimate(self):
        """ Policy Iteration using synchronus backups.
            The goal is to estimate the state value function with a given policy.
            This is an iterative algorithm which is vectorized here.
        """
        # v_0 = 0
        v = np.array([0]*len(self.MDP.all_states))
        # Expected reward R_pi
        n = len(self.MDP.all_states) # number of states
        R_pi = np.zeros((n))
        for i in range(n) :
            R_pi[i] = np.dot(self.reward_matrix[i,:], self.pi_matrix[:,i])
        # transition matrix
        trans_matrix = self.MDP.get_mrp(self.Policy).trans_matrix
        # Iterations
        for i in range(self.max_iter) :
            vk = v
            vk1 = R_pi + self.gamma * np.dot(trans_matrix,vk)
            v = vk1
        # map v to a dict
        d = dict.fromkeys(self.MDP.all_states, 0)
        for i,state in enumerate(d.keys()) :
            d[state] += v[i]
        
        return d
        
            
if __name__ == "__main__" :

    policy_data = {

            1: {'a': 0.4, 'b': 0.6},
            2: {'a': 0.7, 'c': 0.3},
            3: {'b': 1.0}
    }

    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'a': ({1:0.5, 3: 0.5}, 1.0),
            'b': ({2: 0.9, 3:0.1}, 0.0)
        }
    }
    
    a = Policy_Evaluation(data,policy_data,0.5,100)

#    print("a.actions_list = ", a.actions_list,"\n")
#    print("a.reward_matrix = ", a.pi_matrix,"\n")
    print("a.pi_matrix = ", a.pi_matrix ,"\n")
    print("reward matrix = ", a.reward_matrix, "\n")
    print("mrp trans = ", a.MDP.get_mrp(a.Policy).transitions, "\n")
    print("v estimate =" , a, "\n")
    
    
    

