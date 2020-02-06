# README file for CME 241 (Winter quarter 2020)

## Lectures

The _pdf_ files are meant to be a summary of the most important notions studied in lectures. 

* `lecture_1.pdf` sums up  the notions of Markov Chains, Value State Function and Optimal Policies studient during lecture 1. 

## Type of data 

## `./processes` folder

This folder contains Python files for the implementation of Markov Processes, Markov Reward Processes and Markov Decision Processes. All these processes are modelled as Python `class`. Here, the objective is to define objects that will be used in Dynamic Programming and Reinforcement Learning algorithms. The structure of these classes is incremental where `_MP_` class is the basis for all other processes. 

### Usage 

Let us run : 
`python3 mp.py`

Then the output is : 

`· states list :  {1: {2: 0.25, 3: 0.75}, 2: {2: 1}, 3: {2: 0.45, 3: 0.55}} 
· number of states :  3 
· sink states :  {2}
·  matrix transition :  
 [[0.   0.25 0.75]
 [0.   1.   0.  ]
 [0.   0.45 0.55]]
{1: 0.0, 2: 1.0, 3: 0.0}
stationary :  {1: 0.5333333333333333, 2: 0.4666666666666667}`


This folder also contains a Python file `policy.py` for a Policy implementation. This class is used in the class `_MDP_`. 


## `./algorithms` folder

Still in progress 
