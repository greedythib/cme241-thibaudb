""" File for running predictions algorithms.
"""
import sys
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/src/RL")
from MC import MC
from TD import TD
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/src/utils")
from sampling import MDP_sampling
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/src/processes")
from mdp import MDP
from policy import Policy
sys.path.append("/Users/thibaudbruyelle/Documents/Stanford/winter2020/cme241/cme241-thibaudb-master/src/DP")
from policy_evaluation import Policy_Evaluation


if __name__ == "__main__" :

    policy_data = {

            1: {'a': 0.4, 'b': 0.6},
            2: {'a': 0.7, 'c': 0.3},
            3: {'a' : 0.5, 'b': 0.5}
    }

    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 5.0),
            'c': ({1: 0.2, 2: 0.6, 3: 0.2}, -7.2)
        },
        3: {
            'a': ({1:0.5, 3: 0.5}, 1.0),
            'b': ({2: 0.5, 3:0.5}, 10)
        }
    }

    max_ep_length = 100
    ep_num = 1000   # number of episodes in the sequence simulated
    gamma = 0.5       # discount factor


    simul_data = MDP_sampling(data, policy_data, ep_num, max_ep_length ,gamma)

    print("SIMULATION of incomplete MDP sequence with {} episodes".format(ep_num))
    print("-"*80)


    print("Done !")
    print("Maximum size of episodes : {}".format(max_ep_length))
    
    # print infos

    print("-"*80)
    print("PARAMETERS")
    print("gamma = {}".format(gamma))
    

    # REAL STATE VALUE FUNCTION
    print("-"*80)
    my_mdp = MDP(data, gamma)
    print("MDP state value function : ", my_mdp.get_value_func_dict(Policy(policy_data)))
    
    # DP : POLICY EVALUATION
    print("-"*80)
    max_iter = 100
    pol_ev = Policy_Evaluation(data, policy_data, gamma, max_iter)
    print("DP : POLICY EVALUATION with {} iterations.".format(max_iter))
    print("Estimated value function :", pol_ev.get_value_function_estimate())

    # MONTE CARLO model-free prediction
    print("-"*80)
    print("MONTE-CARLO method for model-free prediction")
    MC = MC(simul_data, gamma)
    print("Estimated value function (every visit) :", MC.MC_every_visit())
    
    print("-"*80)
    print("Estimated value function (first visit) :", MC.MC_first_visit())
    print("-"*80)

    # TEMPORAL DIFFERENCE
    print("TEMPORAL-DIFFERENCE(0) method for model-free prediction")
    alpha = 0.01
    print("alpha = {}".format(alpha))
    TD = TD(simul_data, gamma)
    print("Estimated value function :", TD.get_value_function_estimate(alpha))
    print("-"*80)
    
    
    # PLOT 
    
    
    
    
    
