import numpy as np;
import math;
import random as rand;
def sigmoid(v):
    return 1/(1+math.exp(v));
def perceptron(w,x):
    return np.matmul(w,x);
def inputPatternGen():
    pattern = []
    for l in range(0,2):
        for m in range(0,2):
            for n in range(0,2):
                for o in range(0,2):
                    pattern.append(np.array([l,m,n,o]));
    pattern = np.array(pattern);
    return pattern;
def randWeightBias():
    return np.array([rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1)]);
def desiredOutput(inputPattern):
    return sum(inputPattern)%2
