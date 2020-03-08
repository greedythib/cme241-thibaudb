""" Tabular for Q-learning algorithm.

    Goal : Off-policy TD Control
    
"""
import copy
import numpy as np
from control_base import RL_control


class Q_learning(RL_control):
    """
    """
    
    def __init__(self, episodes_data, gamma , q, epsilon, alpha) :
        RL_control.__init__(self, episodes_data, gamma, q)
        self.epsilon = epsilon
        self.alpha = alpha
        
    def argmax_q(self, q, state) :
        """ Helper method to compute the argmax of the state value function with respect
            to its actions (given a state).
            
            @param  q       dict
            @param  state   int
        """
        actions_list = list(q[state].keys())
        arg =actions_list[0]
        max = q[state][arg]
        for action in actions_list :
            if q[state][action] > max :
                max = q[state][action]
                arg = action
        return arg
    
    def get_action_from_greedy_policy(self, q, state) :
        """ Function to choose the next action given a state by using a greedy policy.
        
            @param  q       dict
            @param  state   int
        """
        choice = np.random.choice(["greedy", "random"], p = [1-self.epsilon, self.epsilon])
        if choice == "greedy" :
            return self.argmax_q(q, state)
        else :
            action = np.random.choice(self.actions_list)
            while action not in list(q[state].keys()) :
                action = np.random.choice(self.actions_list)
            return action
        
        
    def get_optimal_state_value_function_estimate(self) :
        """
            Q-Learning algorithm to get the optimal state value function in model-free
            control.
        """
        # 1. Initialization
        q = copy.deepcopy(self.q)
        # 2. Iterations
        for episode in self.episodes_data :
            for k,v in episode.items() :
                # observe the state
                state = k[0]
                t = k[2]
                # 2a. We take action using policy derived from Q :
                action = self.get_action_from_greedy_policy(q,state)
                # 2b. We observe the reward R' and next state S' :
                r = v[1]
                next_state = v[0]
                # 2c. We take the next action A' using S' :
                next_action = self.get_action_from_greedy_policy(q, next_state)
                # 2d. We need to compute max_{action} q(S',action).
                max_val = max(q[next_state].values())
                # 2d. We update the value of q by using Q-Learning update formula :
                q[state][action] += self.alpha * (r + self.gamma * max_val - q[state][action])
                
        return q
                
        
if __name__ == "__main__" :

    data = {
            (1,'a',0) : (1,10),
            (1,'b',1) : (2,-10),
            (2,'a',2) : (2,10),
            (3,'b',3) : (1,0)
           }
           
    data2 = {
            (1,'a',0) : (1,10)
           }
           

    q_init = {
         1 : {'a' : 0, 'b' : 0},
         2 : {'a' : 0, 'c' : 0},
         3 : {'b' : 0 }
        }

    qlearning = Q_learning(episodes_data = [data,data2], gamma = 0.5, q = q_init, epsilon = 0.1, alpha =0.5)


    print("action list = ", qlearning.actions_list)

    print("q = ", qlearning.get_optimal_state_value_function_estimate())

    
    








