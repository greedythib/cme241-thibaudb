from policy import Policy 

class DetPolicy(Policy) : 
    
    def __init__(self, det_policy_data : dict) : 
        
        Policy.__init__(self, {s: {a: 1.0} for s, a in det_policy_data.items()})
        
    def get_action_for_state(self, state) :
        return list(self.get_state_probabilities(state).keys())

    def get_state_to_action_map(self):
        return {s: self.get_action_for_state(s) for s in self.policy_data}

    def __repr__(self) -> str:
        return self.get_state_to_action_map().__repr__()

    def __str__(self) -> str:
        return self.get_state_to_action_map().__str__()

    
if __name__ == "__main__" : 
    policy_data = {
            1: {'a': 0.4, 'b': 0.6},
            2: {'a': 0.7, 'c': 0.3},
            3: {'b': 1.0}
    }
    
    obj = DetPolicy(policy_data)
    
    print("get_action_for_state : ", obj.get_action_for_state(1), '\n')
    print("get_state_to_action_map : ", obj.get_state_to_action_map(), "\n")
    print("__repr__ : ", obj.__repr__(), "\n")
    print("__str__", obj.__str__())
    
        