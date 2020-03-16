""" Temporal-Difference learning in Model-Free prediction.
"""
from prediction_base import RL

class TD(RL) :
    """ Derived class of RL class to implemente Temporal-Difference learning in
        Model-Free prediction.
        
        @param  episodes_data   list   Sequence of episodes of shape dict((state, action) : (next_state, reward))
    """
        
    def get_value_function_estimate(self, alpha = 1) :
        """ Estimation of state value function using TD(0) incremental update.
        
            For each episode, the mean of discounted returns is computed incrementally
            during each episode iteration.
            
            @param  alpha   float
        """
        # Initialization.
        value_function = dict.fromkeys(self.states_list, 0)
        
        # Iterations over all episodes of the random sequences.
        for episode in self.episodes_data :
            # Reward vector.
            R = self.get_reward(episode)
            for k,v in episode.items() :
                state = k[0]
                t = k[2]
                next_state = v[0]
                # We update the value function :
                value_function[state] += alpha*(R[t] + self.gamma*value_function[next_state] - value_function[state])
                
        return value_function
    
    
if __name__ == "__main__" :
# We have an unknown MDP, ie, we do not know transitions and rewards.

    data = {
            (1,'a') : (1,10),
            (1,'b') : (2,-10),
            (2,'a') : (2,10),
            (3,'b') : (1,0)
           }

    test = TD(data, 1)
    
    # SANITY CHECKS
    print("test.data = ",test.data, "\n")
    print("test.rewards = ", test.R_t)
#    print("test.states_list = ", test.states_list, "\n")
#    print("test.incremental_update", test.incremental_update(), "\n")
#    print("test.G_t = ", test.G_t, "\n")
#    print("numbe of steps : ", test.horizon)
    print("value function (TD): ", test.get_value_function_estimate())
