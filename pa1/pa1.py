import numpy as np;
import math;
import random as rand;
def sigmoid(v):
    return 1/(1+math.exp(-v));
def derivSigmoid(v):
    return math.exp(-v)/math.pow(1+math.exp(-v),2)
def perceptron(w,x):
    return np.matmul(w,x);
def inputPatternGen():
    pattern = []
    for l in range(0,2):
        for m in range(0,2):
            for n in range(0,2):
                for o in range(0,2):
                    pattern.append(np.array([1,l,m,n,o]));
    pattern = np.array(pattern);
    return pattern;
def randWeightBias():
    return np.array([rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1),rand.uniform(-1,1)]);
def desiredOutput(inputPattern):
    return sum(inputPattern)%2

n = 0.25;
inputPat = inputPatternGen();
layer1= np.zeros((4,5))
absoluteError = np.zeros(len(inputPat))
endCondition = False
epoch = 0;
for i in range(4):
    layer1[i, 0:5] = randWeightBias();
outputLayer = randWeightBias();
while not endCondition:
    for j in range(len(inputPat)):
        layer1Output = np.zeros(4);
        for i in range(4):
            layer1Output[i] = sigmoid(perceptron(layer1[i],inputPat[j]));
        layer1Output=np.insert(layer1Output,0, [1]);
        forwardPassOutput = sigmoid(perceptron(outputLayer,layer1Output));
        desiredOutput = sum(inputPat[j, 1:5])%2;
        errorOutputLayer = desiredOutput - forwardPassOutput;
        absoluteError[j] = abs(errorOutputLayer);
        localGradientOutputLayer = errorOutputLayer*forwardPassOutput*(1-forwardPassOutput) 
        localGradientLayer1 = np.zeros(4);
        for i in range(4):
            localGradientLayer1[i] = layer1Output[i+1]* (1-layer1Output[i+1]) * localGradientOutputLayer*outputLayer[i+1];
    
        outputLayer = outputLayer + (n * localGradientOutputLayer*layer1Output);
        for i in range(4):
            layer1[i] = layer1[i]+ n *localGradientLayer1[i]*inputPat[j];

    endCondition = True;
    for j in range(len(inputPat)):
        if absoluteError[j] > 0.05:
            endCondition = False;
            print(str(j)+" FAILED.")
    epoch +=1;
    print("n: "+str(n)+", epoch:"+str(epoch));

    