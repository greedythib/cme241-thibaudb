""" Interface for Markov Chains.
"""
import mp_funcs as funcs
import numpy as np 

from scipy.linalg import eig

class MP :
    """ Parent class for MRP, MDP. It enables an incremental structure of these Markov Processes.
    """
    
    def __init__(self, transitions) :
        """ Initialization of a MP.
        
            @param    transitions    dict   Input of shape { state : {state : prob_transition} }
        """
        
        if not funcs.is_mp(transitions) and not funcs.is_mrp(transitions) : 
                
            raise ValueError("wrong transition data format...")
         
        self.transitions = transitions
        self.states_list = [k for k in transitions.keys()]
        self.states_nb = len(transitions)
        self.trans_matrix = self.__get_transition_matrix()
                
    def get_sink_states(self) :
        """ Method function to retrieve the sink states.
        """
        if funcs.is_mp(self.transitions) : 
            return {k for k,v in self.transitions.items() 
                    if (len(v) == 1 and k in v.keys())}
        elif funcs.is_mrp(self.transitions) : 
            return {k for k,v in self.transitions.items() 
                if (len(v[0]) == 1 and k in v[0].keys())}
                    
    def __get_transition_matrix(self) :
        """ Method to retrieve the transition matrix.
        """
        if funcs.is_mp(self.transitions) :  
            P = np.zeros((self.states_nb,self.states_nb), dtype =float)
            for k,v in self.transitions.items() : 
                for i in range(1,self.states_nb+1) : 
                    if i in v.keys() : 
                        P[k-1,i-1] = v[i]
                        
        elif funcs.is_mrp(self.transitions) : 
            P = np.zeros((self.states_nb,self.states_nb), dtype =float)
            for k,v in self.transitions.items() : 
                for i in range(1,self.states_nb+1) : 
                    if i in v[0].keys() : 
                        P[k-1,i-1] = v[0][i]           
        return P 
                
    
    def get_stationary_distribution(self) :
        """
        Returns the stationary distribution of a MP (only if the transition matrix is not)
        singular.
        The package `scipy.linalg` is used for this purpose. 
        """
        eig_vals, eig_vecs = eig(np.transpose(self.trans_matrix))
        stat = np.array(
            eig_vecs[:, np.where(np.abs(eig_vals - 1.) < 1e-8)[0][0]].flat
        ).astype(float)
        norm_stat = stat / sum(stat)
        return {k+1:v for k,v in enumerate(norm_stat)}
        

""" FOR SANITY CHECKS PURPOSES ONLY.
"""

if __name__ == '__main__' :
    
    # We have a finite set of states. 
    states_transition = {1 : {2:0.25,3:0.75},
                         2 : {2:1},
                         3 : {2:0.45,3:0.55}}
                         
    my_mp = MP(states_transition)
    print(" · input data : ", my_mp.transitions,"\n",
          "· number of states : ", my_mp.states_nb, "\n",
         " · sink states : ", my_mp.get_sink_states())
    
    print("matrix transition : ","\n", my_mp.trans_matrix)
    
    stationary = my_mp.get_stationary_distribution()
    print(stationary)
    
    # Example 1
    data = {1 : {1 :0.3,2:0.7},
            2 : {1:0.8,2:0.2},
            }
    
    mp = MP(data)
    print("stationary : ", mp.get_stationary_distribution())
    
