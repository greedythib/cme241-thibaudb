import mp_funcs as funcs

class MRP : 
    
    def __init__(self, transitions, gamma) : 
        
        if  funcs.is_mrp_transitions(transitions) : 
            self.transitions = transitions
            self.states_nb = len(transitions)
            self.gamma = gamma # Discount factor.
        else : 
             raise ValueError("wrong transition matrix format...")
                
    # to do : Create a method to get the stationary distribution of this chain.
    # to do : Create a method to get the state value function of thid chain.
    # to do : Get_Transition Matrix method. 
    
if __name__ == '__main__' : 
    
    # We have a finite set of states. 
    data_mrp = {         1 : ({1:0,2:0.25,3:0.75},10),
                         2 : ({1:1,2:0,3:0},12),
                         3 : ({1:0,2:0.45,3:0.55},-5)}
    my_mrp = MRP(data_mrp, gamma =1)
    print(" · states list : ", my_mrp.transitions,"\n",
          "· number of states : ", my_mrp.states_nb)
