import mp_funcs as funcs

class MP : 
    
    def __init__(self, transitions) : 
        
        if funcs.is_mp_transitions(transitions) : 
            self.transitions = transitions
            self.states_nb = len(transitions)
        else : 
             raise ValueError("wrong transition matrix format...")
                
    # to do : Create a method to get the stationary distribution of this chain. 
    
if __name__ == '__main__' : 
    
    # We have a finite set of states. 
    states_transition = {1 : {1:0,2:0.25,3:0.75},
                         2 : {1:1,2:0,3:0},
                         3 : {1:0,2:0.45,3:0.55}}
    my_mp = MP(states_transition)
    print(" · states list : ", my_mp.transitions,"\n",
          "· number of states : ", my_mp.states_nb)
