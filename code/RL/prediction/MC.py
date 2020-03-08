""" Monte-Carlo learning in Model-Free prediction. 
"""
from prediction_base import RL
import numpy as np

class MC(RL) :
    """ Derived class of RL class to implemente Monte-Carlo learning in
        Model-Free prediction. MC requires complete episodes. 
        
        @param    episodes_data    list    List of episodes of type dict((state, action, time_step) : (next_state, reward)) Sequence of episodes
    """
        
    def get_value_function_estimate(self, method ,alpha = None) :
        """ Estimation of state value function using Monte-Carlo incremental update.
           
            @param[in]  method  str     Whether every-visit or first-visit method.
            @param[in]  alpha   float   Learning rate between [0,1].
        """
        
        if method == "every_visit" :
            return self.MC_every_vist(alpha)
        elif method == "first_visit" :
            return self.MC_first_visit()
        else :
            return "The method given is not valid"
        

    def MC_every_visit(self, alpha = None) :
        """ Every-visit MC algorithm for model-free prediction.
            
            The mean of discounted returns is computed incrementally over
            all the episodes.
            
            @param[in]  alpha   float   Learning rate between [0,1].
            
            Returns the estimated value function.
        """
        # Initialization
        counts = dict.fromkeys(self.states_list,0)
        value_function = dict.fromkeys(self.states_list,0)
        
        # Iterations (whether alpha is set)
        for episode in self.episodes_data :
            G = self.get_return(episode)
            if alpha == None :
                for k,v in episode.items() :
                    state = k[0]
                    t = k[2]
                    counts[state] += 1
                    v = value_function[state]
                    value_function[state] += (G[t] - v)*(1/counts[state])
            else :
                for k,v in episode.items() :
                    state = k[0]
                    time_step = k[2]
                    counts[state] += 1
                    v = value_function[state]
                    value_function[state] += (G[time_step] - v)*alpha
           
        return value_function
        
    def MC_first_visit(self) : # Not optimal yet
        """ First-visit MC algorithm for model-free prediction
        
            When a step is visited or the first time during an episode,
            the discounted return is added to its returns list. Then the mean is
            computed for all states at the end of all episodes.
        """
        # Initialization
        value_function = dict.fromkeys(self.states_list,0)
        returns_S = {k:[] for k in self.states_list}

        # Iterations (whether alpha is set)
        for episode in self.episodes_data :
            detect_first_visit = dict.fromkeys(self.states_list,0)
            G = self.get_return(episode)
            for k,v in episode.items() :
                state = k[0]
                t = k[2]
                if detect_first_visit[state] == 0 : # first visit of the state
                    returns_S[state].append(G[t])
                detect_first_visit[state] +=1
                
        for s in self.states_list :
            if len(returns_S[s]) != 0 :
                value_function[s] += np.mean(returns_S[s])
                
        return value_function
                
                
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

    
