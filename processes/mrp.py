import mp_funcs as funcs
from mp import MP 

class MRP(MP) : 
    
    def __init__(self, transitions) : 
        
        MP.__init__(self, transitions)
        
    
                
    # to do : Create a method to get the stationary distribution of this chain.
    # to do : Create a method to get the state value function of thid chain.
    # to do : Get_Transition Matrix method. 
    
if __name__ == '__main__' : 
    
    # We have a finite set of states. 
    data_mrp = {1 : ({1:0,2:0.25,3:0.75},10),
                2 : ({2:1},15),
                3 : ({1:0,2:0.45,3:0.55},-5)}
    my_mrp = MRP(data_mrp)
    print(my_mrp.get_sink_states())
    print(" · states list : ", my_mrp.transitions,"\n",
          "· number of states : ", my_mrp.states_nb, "\n",
         " · sink states : ", my_mrp.get_sink_states())
    
    # print("matrix transition : ","\n", my_mrp.trans_matrix)
    
    


