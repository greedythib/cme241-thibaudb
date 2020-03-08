import matplotlib.pyplot as plt
import numpy as np 


# Number of lilypads. 
n = 25

# Transition Matrix 
## for action A 
P_A = np.zeros((n+1,n+1))
P_A[0,0] = 1
P_A[n,n] = 1

for i in range(1,n) : 
    P_A[i,i+1] = (n-i)/n
    P_A[i,i-1] = i/n
    
## for action B 
P_B = np.zeros((n+1,n+1))
P_B[0,0] = 1
P_B[n,n] = 1

for i in range(1,n) :
    P_B[i,:] = 1/n
    P_B[i,i] = 0 
    
    
# Reward Function 
## for action A 
R_A = np.zeros(n+1)
R_A[0] = -1
R_A[1] = -1/n
R_A[n-1] = 1/n
R_A[n] = +1

## for action B 
R_B = np.zeros(n+1)
R_B[0] = -1
R_B[n] = +1

# Discount factor 
gamma = 0.9

# Initialization of v0, the state value function.
# We avoid the null reward in state 1 due to action A. 
v = np.zeros(n+1)
v[0] = -1
v[n] = 1

opti_policy = ['A']*(n+1)
opti_policy[1] = 'B'


# Value Iteration
## Number of iterations 
K = 1000
## For loop. 
for k in range(K) : 
    v_old = v.copy()
    v_new = v.copy()
    for s in range(len(v_old)) : 
        val_A = R_A[s] + gamma*np.dot(P_A[s,:], v_old)
        val_B = R_B[s] + gamma*np.dot(P_B[s,:], v_old)
        v_new[s] = max(val_A, val_B)
        if val_A > val_B : 
            opti_policy[s] = 'A'
        else : 
            opti_policy[s] = 'B'
    # We update the vector.         
    v = v_new
    
print("v = ", v)
print("Optimal Policy : ", opti_policy)


# Let us compute the optimal escape probability. 

## First, we need to get the Markov Chain transitions using the Optimal Policy 
P = np.zeros((n+1,n+1))
for i in range(len(opti_policy)) : 
    if opti_policy[i] == 'A' : 
        P[i,:] = P_A[i,:]
    else : 
        P[i,:] = P_B[i,:]
        
print(P)

I = np.eye(n+1)
I[0,0] = 0
I[n,n] = 0 
b = np.zeros(n+1)
b[n] = 1  

opt_esc_prob = np.linalg.solve(P-I,b)
print(opt_esc_prob)

# Plot 
X=[i for i in range(n+1)]
plt.bar(X,opt_esc_prob)

for i in range(len(X)):
    plt.annotate(opti_policy[i], xy=(X[i],opt_esc_prob[i]))    
plt.show()



