""" Interface for Policy.
"""
import mp_funcs as funcs

class Policy :
    """ Policy Class.
        Used in MRP methods (e.g. to compute the State-Value function)
    """
    
    def __init__(self, data : dict) :
        """ Initialization of the Policy.
            
            @param    data    dict    Input of shape {state : {action : prob_2_take_action}}
        """
        
        if funcs.is_policy(data) : 
            self.policy_data = data 
        else : 
            raise ValueError("Wrong policy data !")
            
    def get_state_probabilities(self, state) :
        return self.policy_data[state]

    def get_state_action_probability(self, state, action) -> float:
        """ Method to retrieve the probability of taking an action given a state.
        """
        return self.get_state_probabilities(state).get(action, 0.)

    def edit_state_action_to_epsilon_greedy(
        self,
        state,
        action_value_dict,
        epsilon: float
    )  :
        self.policy_data[state] = get_epsilon_action_probs(
            action_value_dict,
            epsilon
        )        
            
    def __str__(self):
        return self.policy_data.__str__()
    
""" FOR SANITY CHECKS PURPOSES ONLY. 
"""
if __name__ == "__main__" : 
    policy_data = {
            1: {'a': 0.4, 'b': 0.6},
            2: {'a': 0.7, 'c': 0.3},
            3: {'b': 1.0}
    }
    
    my_policy = Policy(policy_data)
    print("my_policy state action prob : ", my_policy.get_state_action_probability(1,"c"), "\n")
    
    print(str(my_policy))
