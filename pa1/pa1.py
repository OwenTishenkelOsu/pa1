import numpy as np;
import math;
import random as rand;
import matplotlib.pyplot as plt
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

n = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
epoch = np.zeros(10);
meanSquareError = [[] for val in n]
inputPat = inputPatternGen();
for y in range(len(n)):
    layer1= np.zeros((4,5))
    absoluteError = np.zeros(len(inputPat))
    endCondition = False
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
            localGradientOutputLayer = errorOutputLayer* derivSigmoid(perceptron(outputLayer,layer1Output))
            #errorOutputLayer*forwardPassOutput*(1-forwardPassOutput) 
            localGradientLayer1 = np.zeros(4);
            
            for i in range(4):
                localGradientLayer1[i] = derivSigmoid(perceptron(layer1[i],inputPat[j])) * localGradientOutputLayer*outputLayer[i+1]
                #layer1Output[i+1]* (1-layer1Output[i+1]) * localGradientOutputLayer*outputLayer[i+1];
            outputLayer = outputLayer + (n[y] * localGradientOutputLayer*layer1Output);
            
            for i in range(4):
                layer1[i] = layer1[i]+ n[y] *localGradientLayer1[i]*inputPat[j];

        endCondition = True;
        meanError = 0;
        for j in range(len(inputPat)):
            meanError = meanError+absoluteError[j];
            if absoluteError[j] > 0.05:
                endCondition = False;
                print(str(j)+" FAILED. Error:"+str(absoluteError[j]))
        meanError = meanError/ len(inputPat);
        meanSquareError[y].append(meanError)
        epoch[y] +=1;
        print("Mean Error:"+str(meanError));
        print("n: "+str(n[y])+", epoch:"+str(epoch[y]));
print("N: "+str(n)+"\n Epochs:"+str(epoch));
for val in range(len(n)):
    plt.figure(val);
    plt.plot(meanSquareError[val]);
    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Mean Square Error')
  
    # giving a title to my graph
    plt.title('Learning Curve of '+str(n[val]))
  
    # function to show the plot
    plt.show()

    