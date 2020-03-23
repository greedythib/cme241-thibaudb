""" CONTROL : We want to estimate the optimal policy given a MDP.
    Interface for POLICY ITERATION algorithm.
"""

from base import DP
import numpy as np
import copy
from policy_evaluation import Policy_Evaluation

class Policy_Iteration(DP) :
    """ Derived class of DP class to implemente Policy Iteration algorithm.
        It uses Iterative Policy Evaluation Algorithm.
    """
    
    def get_value_function_estimate(self) :
        return "Wrong instance : object is for control (policy iteration)"
        
    def argmax_v(self, v, state) :
        """ Helper method to compute the argmax of the state value function with respect
            to its actions (given a state).
            
            @param  v       dict
            @param  state   int
        """
        actions_list = list(v[state].keys())
        arg =actions_list[0]
        max = v[state][arg]
        for action in actions_list :
            if v[state][action] > max :
                max = v[state][action]
                arg = action
        return arg
      
      
    def get_optimal_value_function_estimate(self):
        """ Policy Iteration using synchronus backups.
            The goal is to estimate the optimal state value function.
            
            Returns the optimal State-Value function, Action-Value function and
            Optimal policy.
        """
        
        # 1. Initialization
        ## value function
        value_function = {state : 0 for state in self.MDP.all_states}
#        print(value_function)
        ## random policy
        policy = {state:{} for state in self.MDP.all_states}
        for state in self.MDP.all_states :
            for action in self.mdp_data[state].keys() :
                policy[state][action] = 0
        ### I chose to take a deterministic policy by given weight 1 to the first action
        ### for each state.
        for state in self.MDP.all_states :
            for action in list(policy[state].keys())[1] :
                policy[state][action] = 1
        print(policy)
        
        # 2. Iterations
        for iter in range(10) :
            
            # 2.1 Policy Evaluation with 100 iterations
            pol_eval = Policy_Evaluation(self.mdp_data, policy, self.gamma, 100)
            
#            print("pol_eval.val_func", pol_eval.get_value_function_estimate())
            value_function = pol_eval.get_value_function_estimate()
            action_value_function = pol_eval.get_action_value_function()
#            print("action value =", action_value_function)
            
            # 2.2 Policy Improvement
            policy_stable : bool = True
            for state in self.MDP.all_states :
                for action in policy[state].keys() :
                    if policy[state][action] == 1 :
                        old_action = action
                
                pi_state = self.argmax_v(action_value_function,state)
                
                if pi_state != old_action :
                    policy_stable = False
                    # In this case we need to update the policy by imporving it greedily.
                    policy[state][old_action] = 0
                    policy[state][pi_state] = 1
                    
#            # Stop condition
#            if policy_stable == True :
#                return value_function, action_value_function
#            else :
#                continue
          
        res_policy = {state:None for state in self.MDP.all_states}
        for state in self.MDP.all_states :
            for action in self.mdp_data[state].keys() :
                if policy[state][action] == 1 :
                    res_policy[state] = action
            
        return value_function, action_value_function, res_policy
        
        
    def get_optimal_action_value_function(self) :
        return 0
        
    
        
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
    
    a = Policy_Iteration(data,policy_data,0.5,100)

    print("v : ", a.get_optimal_value_function_estimate()[0])
    print("q : ", a.get_optimal_value_function_estimate()[1])
    print("policy : ", a.get_optimal_value_function_estimate()[2])
    
    
    

