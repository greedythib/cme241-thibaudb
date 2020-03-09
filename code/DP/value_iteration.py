""" CONTROL : We want to find a MDP optimal policy.
    Interface for VALUE ITERATION algorithm.
    For this purpose, we make an iterative application of Bellman optimality equations.
"""
import numpy as np
from base import DP

class Value_Iteration(DP) :
    """ Derived class of DP class to implement Value Iteration algorithm.
    """
    
    def get_value_function_estimate(self) :
        return "Wrong instance : object is for control (value iteration)"
    
    def get_optimal_value_function_estimate(self):
        """ Value Iteration using synchronus backups.
            The goal is to estimate the optimal state value function.
            This is an iterative algorithm which is vectorized.
        """
        # v_0 = 0
        v = np.array([0]*len(self.MDP.all_states), dtype = float)
        # Reward matrix
        R_mat = self.reward_matrix
        
        for iter in range(self.max_iter) :
            v_old = v.copy()
            v_new = v.copy()

            # Iteration over each state (we need to maximize with respect to the actions)
            for state in self.MDP.all_states :
                # Consider the  reward vector when in `state` with `action`.
                qty_to_max = R_mat[state -1,:].copy() # row vector of size number_of_actions
                
                # We add the future contribution terms.
                for idx, action in enumerate(self.actions_list) :
                    P_s_a = self.get_trans_matrix_on_action(action)[state-1,:]

                    qty_to_max[idx] += self.gamma * np.dot(P_s_a, v_old)

                # We need to chose the action that maximizes `qty_to_max`.
                argmax_idx = np.argmax(qty_to_max)

                v_new[state-1] = qty_to_max[argmax_idx]

            # Update the value.
            v = v_new.copy()
            
            
        value_function = {state:0 for state in self.MDP.all_states}
        for idx,s in enumerate(value_function.keys()) :
            value_function[s] = v[idx]
        
        return value_function
        
        
    def get_optimal_action_value_function(self) :
        """ Returns the optimal action value function by using the optimal value
            function. It uses the Bellman Optimality Equation. 
        """
        v_opt = np.array([0]*len(self.actions_list), dtype = float)
        for k,v in self.get_optimal_value_function_estimate().items() :
            v_opt[k-1] = v
        R_mat = self.reward_matrix
        # Initialization of the optimal value function.
        opt_action_vf = {state:{} for state in self.MDP.all_states}
        for state, dict in self.mdp_data.items() :
            for action in dict.keys() :
                opt_action_vf[state][action] = 0
        # Update the values by using the Bellman optimality equations.
        for state,dict in opt_action_vf.items() :
        
            for idx,action in enumerate(dict.keys()) :
                opt_action_vf[state][action] += R_mat[state -1,idx]
                for s in self.MDP.all_states :
                    P_s_a = self.get_trans_matrix_on_action(action)[s-1,:]
                    opt_action_vf[state][action] +=  self.gamma * np.dot(P_s_a, v_opt)
                
        return opt_action_vf
        
""" FOR SANITY CHECKS PURPOSES ONLY.
"""
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
    
    a = Value_Iteration(data,policy_data,0.05,10)

#    print("a.actions_list = ", a.actions_list,"\n")
#    print("a.reward_matrix = ", a.pi_matrix,"\n")
    print("a.pi_matrix = ", a.pi_matrix ,"\n")
    print("reward matrix = ", a.reward_matrix, "\n")
#    print("mrp trans = ", a.MDP.get_mrp(a.Policy).transitions, "\n")
#    print("v estimate =" , a, "\n")


    print("states list : ", a.MDP.all_states)
    print("trans_matrix =",a.get_trans_matrix_on_action('b'))
    
    print("gamma : ", a.gamma)
    
    print("optimal_value_function_estimate :", a.get_optimal_action_value_function())
    

    
    
    

