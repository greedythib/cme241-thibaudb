import math

def is_mp(transitions) :
    states_nb = len(transitions)
    for k in transitions.keys() :
        if type(transitions[k])==tuple : 
            return False 
        prob_sum = 0 
        for state in transitions[k] : 
            prob_sum += transitions[k][state]
        if prob_sum != 1 : 
            return False
    return True

def is_mrp(transitions):
    states_nb = len(transitions)
    for k in transitions.keys() : 
        if type(transitions[k])!=tuple : 
            return False 
        prob_sum : int = 0  
        for state in transitions[k][0].keys() : 
            prob_sum += transitions[k][0][state]
        if math.ceil(prob_sum) != 1.0 :
            return False 
    return True  

def is_policy(data) :
    if type(data) != dict : 
        return False 
    for state in data.keys() : 
        if type(data[state]) != dict : 
            return False 
    return True 


