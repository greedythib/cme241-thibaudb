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

"""
Useful functions to manipulate MDP. 
"""

def get_actions_for_states(mdp_data):
    return {k: set(v.keys()) for k, v in mdp_data.items()}

def mdp_rep_to_mrp_rep1(
    mdp_rep,policy_rep) : 
    return {s: sum_dicts([{s1: p * v2 for s1, v2 in v[a].items()}
                          for a, p in policy_rep[s].items()])
            for s, v in mdp_rep.items()}
def mdp_rep_to_mrp_rep2(
    mdp_rep,
    policy_rep):
    return {s: sum(p * v[a] for a, p in policy_rep[s].items())
            for s, v in mdp_rep.items()}

def sum_dicts(dicts):
    return {k: sum(d.get(k, 0) for d in dicts)
            for k in set.union(*[set(d1) for d1 in dicts])}
