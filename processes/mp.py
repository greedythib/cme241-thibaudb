import mp_funcs as funcs
import numpy as np 

class MP : 
    
    def __init__(self, transitions) : 
        
        self.transitions = transitions
        self.states_nb = len(transitions)
        
        if funcs.is_mp_transitions(transitions) : 
            self.trans_matrix = self.__get_transition_matrix()
        else : 
             raise ValueError("wrong transition data format...")
                
    def get_sink_states(self) :
        
        if funcs.is_mp_transitions(self.transitions) : 
            return {k for k,v in self.transitions.items() 
                if (len(v) == 1 and k in v.keys())}
        elif funcs.is_mrp_transitions(self.transitions) : 
            return {k for k,v in self.transitions.items() 
                if (len(v[0]) == 1 and k in v[0].keys())}
                    
    
    def __get_transition_matrix(self) : 
        P = np.zeros((self.states_nb,self.states_nb), dtype =float)
        for k,v in self.transitions.items() : 
            for i in range(1,self.states_nb+1) : 
                if i in v.keys() : 
                    P[k-1,i-1] = v[i]
        return P 
                
    
    # def get_stationary_distribution(self) : 
                
    # to do : Create a method to get the stationary distribution of this chain. 
    
if __name__ == '__main__' : 
    
    # We have a finite set of states. 
    states_transition = {1 : {1:0,2:0.25,3:0.75},
                         2 : {2:1},
                         3 : {1:0,2:0.45,3:0.55}}
    my_mp = MP(states_transition)
    print(" · states list : ", my_mp.transitions,"\n",
          "· number of states : ", my_mp.states_nb, "\n",
         " · sink states : ", my_mp.get_sink_states())
    
    print("matrix transition : ","\n", my_mp.trans_matrix)


