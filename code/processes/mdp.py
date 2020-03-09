""" Interface for Markov Decision Processes.
"""
from mp import MP
from mp_funcs import get_actions_for_states, mdp_rep_to_mrp_rep1, mdp_rep_to_mrp_rep2, sum_dicts
from mrp import MRP
from policy import Policy
from det_policy import DetPolicy
# from utils.functions import zip_dict_of_tuple 

""" Helper functions.
"""
def zip_dict_of_tuple(d) :
    """ Helper function for MDP class methods and attributes.
    """
    d1 = {k:v1 for k,(v1,_) in d.items()}
    d2 = {k:v2 for k,(_,v2) in d.items()}
    return d1,d2


class MDP(MRP) :
    """ Derived class of MRP.
    """
    
    def __init__(self, data, gamma : float) -> None :
        """ Initialization of a MRP using MP class a parent class.
        
            @param    data    dict     Input of shape { state : { action : ({state : prob_transition}, reward) } }
            @param    gamma   float    Discount factor.
        """
        
        # Preprocess
        d = {k: zip_dict_of_tuple(v) for k,v in data.items()}
        d1, d2 = zip_dict_of_tuple(d)
        
        # Defining the attributes. 
        self.all_states = set(d1.keys())
        self.state_action_dict = get_actions_for_states(data)
        self.transitions = d1
        self.rewards = d2
        self.gamma = gamma
        self.terminal_states = self.get_terminal_states()
        
        #MRP.__init__(self, d1, gamma)
        
    def get_sink_states(self) :
        """ Method to retrieve the sink states.
        """
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }  
    
    def get_terminal_states(self):
        """
        Overrides the MP method since the input data is different.
        A terminal state is a sink state (100% probability to going back
        to itself, FOR EACH ACTION) and the rewards on those transitions back
        to itself are zero.
        """
        sink = self.get_sink_states()
        return {s for s in sink if
                 all(r==0 for _, r in self.rewards[s].items())}
    
    def get_mrp(self, pol: Policy) -> MRP:
        """ Method to compute the MRP from the MDP given a Policy.
        """
        transitions = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rewards = mdp_rep_to_mrp_rep2(self.rewards, pol.policy_data)
        return MRP({s: (v, rewards[s]) for s, v in transitions.items()}, self.gamma)
    
    def get_value_func_dict(self, pol: Policy):
        """ Method to compute the exact State-Value Function of the MRP.
        """
        mrp_obj = self.get_mrp(pol)
        value_func_vec = mrp_obj.get_state_value_function()
        nt_vf = {mrp_obj.states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.states_list))}
        t_vf = {s: 0. for s in self.terminal_states}
        return {**nt_vf, **t_vf}
    
    def get_act_value_func_dict(self, pol: Policy):
        """ Method to compute the exact Action-Value Function of the MDP.
        """
        v_dict = self.get_value_func_dict(pol)
        return {s: {a: r + self.gamma * sum(p * v_dict[s1] for s1, p in
                                            self.transitions[s][a].items())
                    for a, r in v.items()}
                    for s, v in self.rewards.items()}


""" FOR SANITY PURPOSES ONLY.
"""

if __name__ == '__main__':
   
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
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'a': ({3: 1.0}, 0.0),
            'b': ({3: 1.0}, 0.0)
        }
    }
    mdp_obj = MDP(data, 0.95)
    
   
    print(mdp_obj.get_mrp(Policy(policy_data)).get_state_value_function(), "\n")
    print(mdp_obj.get_act_value_func_dict(Policy(policy_data)))

    #print("rewards1 : ", mdp_obj.rewards_vec, "\n")
    #print("rewards2 : ", mdp_obj.rewards2, "\n")
    
    
    
