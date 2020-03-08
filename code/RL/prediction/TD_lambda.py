""" Temporal-Difference Lambda algorithms for model-free prediction
"""
from tabular import RL

class TD_lambda(RL) :
    """ Derived class of RL class to implemente Monte-Carlo learning in
        Model-Free prediction.
        
        @param episode_data dict((state, action) : (next_state, reward)) Sequence of episodes
    """
    
    def is_state_equ2_s(self,state,s) :
        """ Helper function to compute eligibility trace value.
            
            @param  state   int   Number of the current state (e.g 1)
            @param  s       int   Number of a state within `self.states_list`
            
            Returns a boolean whether the states are equal.
        """
        if state == s :
            return 1
        else :
            return 0
    
    def forward_view_TD_lambda(self, lambda_par, alpha = 0.5) :
        """ Estimation of state value function using TD incremental update.
            The updates are online.
        """
        # Initialization.
        value_function = dict.fromkeys(self.states_list, 0)
        # Iterations.
        for episode in self.episodes_data :
            for k,v in episode.items() :
                state = k[0]
                t = k[2]
                # We compute lambda-returns.
                G_lambda_t = self.get_lambda_return(lambda_par, episode, value_function)
                # We update the value function :
                value_function[state] += alpha*(G_lambda_t[t] - value_function[state])
        return value_function
    
    def backward_view_TD_lambda(self, lambda_par, alpha = 0.5) :
        """ Eligibility Traces algorithm for value function prediction.
            The updates are online.
        """
        # Initialization.
        value_function = dict.fromkeys(self.states_list, 0)
        # Iterations.
        for episode in self.episodes_data :
            # Get reward vector
            R = self.get_reward(episode)
            # Initialize eligibility trace
            E_tr = dict.fromkeys(self.states_list,[0]*self.horizon)
            for k,v in episode.items() :
                state = k[0]
                next_state = v[0]
                t=k[2]
                # Get Temporal-Difference Error
                TD_error = R[t] + (self.gamma)*value_function[next_state] - value_function[state]
                # Get eligibility trace
                # Update all the eligibility trace vector for all states
                for s in self.states_list :
                    if t>0 :
                        E_tr[s][t] = self.gamma * lambda_par * E_tr[s][t-1] + self.is_state_equ2_s(state,s)
                    else :
                        E_tr[s][t] += self.is_state_equ2_s(state,s)
                # Incremental (online update)
                value_function[state] += alpha*TD_error*E_tr[state][t]
        return value_function
            
    def get_value_function_estimate(self, method, alpha = 1, lambda_par =0.5) :
        """ Method TD(lambda) for model-free prediction.
        
            @param  method  str    forward or backward view for TD(lambda) method.
        """
        if method == "forward" :
            return self.forward_view_TD_lambda(lambda_par, alpha)
        elif method == "backward" :
            return self.backward_view_TD_lambda(lambda_par, alpha)
        else :
            return "Not a valid method. "
    
    
if __name__ == "__main__" :
# We have an unknown MDP, ie, we do not know transitions and rewards.

    data = {
            (1,'a',0) : (1,10),
            (1,'b',1) : (2,-5),
            (2,'a',2) : (2,10),
            (3,'b',3) : (1,0)
           }

    test = TD_lambda([data], 0.5)
    
    # SANITY CHECKS
#    print("test.data = ",test.data, "\n")
#    print("test.rewards = ", test.R_t)
##    print("test.states_list = ", test.states_list, "\n")
##    print("test.incremental_update", test.incremental_update(), "\n")
##    print("test.G_t = ", test.G_t, "\n")
##    print("numbe of steps : ", test.horizon)
#    print("value function (TD): ", test.get_value_function_estimate())

    val_estimate = {1 : 1, 2 : 1, 3 : 1}
    print("reward = ", test.get_reward(data))
    
#    print("G_lmbda =", test.get_lambda_return(0.5, data, val_estimate))
    
    print("test.get_value_function_estimate(forward, alpha = 0.1) = ", test.get_value_function_estimate("forward", alpha = 0.1))
    
    print("test.get_value_function_estimate(backward, alpha = 0.1) = ", test.get_value_function_estimate("backward", alpha = 0.1))

