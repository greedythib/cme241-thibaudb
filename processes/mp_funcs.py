def is_mp_transitions(transitions) :
    res = True
    states_nb = len(transitions)
    for k in transitions.keys() : 
        prob_sum = 0 
        for state in transitions[k] : 
            prob_sum += transitions[k][state]
        if prob_sum != 1 : 
            res = False
    return res

def is_mrp_transitions(transitions):
    res = True 
    states_nb = len(transitions)
    for k in transitions.keys() : 
        if len([transitions[k][1]]) != 1 : 
            res = False 
        prob_sum = 0 
        for state in transitions[k][0].keys() : 
            prob_sum += transitions[k][0][state]
        if prob_sum != 1 : 
            res = False
    return res 


