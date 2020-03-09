""" Interface for Markov Reward Processes.
"""
import mp_funcs as funcs
import numpy as np
from mp import MP 

class MRP(MP) :
    """ Derived class of MP.
    """
    
    def __init__(self, transitions, gamma : float) :
        """ Initialization of a MRP using MP class a parent class. 
        
            @param    transitions    dict     Input of shape { state : ({state : prob_transition}, reward) }
            @param    gamma          float    Discount factor.
        """
        
        MP.__init__(self, transitions) # inheritance of MP class.
        self.rewards_vec = self.__get_rewards_vec()
        self.gamma = gamma
    
    def __get_rewards_vec(self) :
        """ Method to retrieve the rewards per state.
        """
        return np.array([self.transitions[k][1] for k in self.transitions.keys()])
    
    def get_state_value_function(self) :
        """
            Method to compute the exact state value function.
            The matrix might be singular.
        """
        try :
        
            return np.linalg.inv(
                   np.eye(self.states_nb) - self.gamma * self.trans_matrix
                   ).dot(self.rewards_vec)
        except np.linalg.LinAlgError :
            print("Singular matrix ! Checkout the terminal states ! ")
        
            
""" FOR SANITY CHECKS PURPOSES ONLY.
"""
if __name__ == '__main__' : 
    
    # We have a finite set of states. 
    data_1 = {
              1 : ({2:0.25,3:0.75},10),
              2 : ({2:1},15),
              3 : ({2:0.45,3:0.55},-5)}
    
    data_2 = {
              1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
              2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
              3: ({3: 1.0}, 0.0)
    }
    
    my_mrp = MRP(data_2, gamma = 1)

    print(" · input data : ", my_mrp.transitions,"\n",
          "· number of states : ", my_mrp.states_nb, "\n",
         " · sink states : ", my_mrp.get_sink_states())
    
    print("matrix transition : ","\n", my_mrp.trans_matrix,"\n")
    print("Rewards vector : ","\n", my_mrp.rewards_vec, "\n")
    
    print("state value function : ", my_mrp.get_state_value_function())
    
    
    
