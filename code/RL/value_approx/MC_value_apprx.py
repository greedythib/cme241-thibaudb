""" Monte-Carlo learning in Model-Free prediction with Value Function approximation.
"""
from value_approx_base import Value_Function_Approximation
import numpy as np

class MC(Value_Function_Approximation) :
    """ Derived class of RL class to implemente Monte-Carlo learning in
        Model-Free prediction. MC requires complete episodes. 
        
        @param    episodes_data    list    List of episodes of type dict((state, action, time_step) : (next_state, reward)) Sequence of episodes
    """
        
    def get_value_function_estimate(self, method ,alpha = 0.1) :
        """ Gradient Monte Carlo Algorithm for Estimating the State Value function.
           
            @param[in]  alpha   float   Learning rate between [0,1].
        """
        # Initialization
        w = np.array([0]*len(self.states_list), dtype = float)
        for episode in self.episodes_data :
            G = self.get_return(episode)
            for k in episode.keys() :
                state = k[0]
                t = k[3]
                # Compute the gradient (vector of shape [0,...0,1,0,...0])
                grad = np.array([0]*len(self.states_list))
                grad[self.states2idx[state]] = 1
                # Get the estimate of the state-value function.
                v_estimate = self.get_state_value_function_approx(w)
                # Update w
                w += alpha*(G[t] - v_estimate[self.states2idx[state]])*grad
         
        # Then we return the estimate of v_pi :
        return self.get_state_value_function_approx(w)
            
    
if __name__ == "__main__" :

# We have an unknown MDP, ie, we do not know transitions and rewards.

    data = {
            (1,'a',0) : (1,10),
            (1,'b',1) : (2,-10),
            (2,'a',2) : (2,10),
            (3,'b',3) : (1,0)
           }
           
    data2 = {
            (4,'a',0) : (1,10)
           }
           

    test = MC([data,data2], 0.5)
    
    # SANITY CHECKS

    print("MC_every_vist : ", test.MC_every_visit())

    

