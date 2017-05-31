# Pre-requisites
import numpy as np
import time

# Initializing weight matrices from layer sizes


def initializeWeights(layers):
    weights = [np.random.randn(o, i + 1)
               for i, o in zip(layers[:-1], layers[1:])]
    return weights

# Add a bias term to every data point in the input


def addBiasTerms(X):
    # Make the input an np.array()
    X = np.array(X)

    # Forcing 1D vectors to be 2D matrices of 1xlength dimensions
    if X.ndim == 1:
        X = np.reshape(X, (1, len(X)))

    # Inserting bias terms
    X = np.insert(X, 0, 1, axis=1)

    return X

# Sigmoid function


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# Forward Propagation of outputs


def forwardProp(X, weights):
    # Initializing an empty list of outputs
    outputs = []

    # Assigning a name to reuse as inputs
    inputs = X

    # For each layer
    for w in weights:
        # Add bias term to input
        inputs = addBiasTerms(inputs)

        # Y = Sigmoid ( X .* W^T )
        outputs.append(sigmoid(np.dot(inputs, w.T)))

        # Input of next layer is output of this layer
        inputs = outputs[-1]

    return outputs

# Compute COST (J) of Neural Network


def nnCost(weights, X, Y):
    # Calculate yPred
    yPred = forwardProp(X, weights)[-1]

    # Compute J
    J = 0.5 * np.sum((yPred - Y)**2) / len(Y)

    return J

# Evaluate the accuracy of weights for input X and desired outptut Y


def evaluate(weights, X, Y):
    yPreds = forwardProp(X, weights)[-1]
    # Check if maximum probability is from that neuron corresponding to desired class,
    # AND check if that maximum probability is greater than 0.5
    yes = sum(int((np.argmax(yPreds[i]) == np.argmax(Y[i])) and
                  ((yPreds[i][np.argmax(yPreds[i])] > 0.5) == (Y[i][np.argmax(Y[i])] > 0.5)))
              for i in range(len(Y)))
    return yes


# TRAINING USING MINI-BATCH GRADIENT DESCENT
def trainUsingMinibatchGD(weights, X, Y, minibatchSize, nEpochs, learningRate=1.0,
                          decay=None, decayRate=0.1, optimizer=None, mu=0.9, testX=None, testY=None):
    # If testing data is not provided, check accuracy on training data
    if testX is None:
        testX = X
        testY = Y

    # Check cost and accuracy
    # Initialize cost
    prevCost = nnCost(weights, X, Y)
    yes = evaluate(weights, X, Y)
    print("Before training: " + str(yes) + " of " + str(len(Y)) + " = " + str(round(float(yes / len(Y)), 4)) +
          "; cost=" + str(prevCost))

    if testY is not Y:
        testPrevCost = nnCost(weights, testX, testY)
        testYes = evaluate(weights, testX, testY)
        print("  Val: " + str(testYes) + " of " + str(len(testY)) + " = " + str(round(float(testYes / len(testY)), 4)) +
          "; val cost=" + str(testPrevCost))

    # Backup weights to revert back in case cost increases
    oldWeights = [np.array(w) for w in weights]

    # To count the number of times learning rate had to be halved contiguously
    countLRHalf = 0

    # Initialize index for iteration through epochs
    epoch = 0

    # For nEpochs number of epochs:
    while epoch < nEpochs:
        # clear output
        # clear_output()

        # Make a list of all the indices
        fullIdx = list(range(len(Y)))

        # Shuffle the full index
        np.random.shuffle(fullIdx)

        # Count number of mini-batches
        nOfMinibatches = int(len(X) / minibatchSize)

        # For each mini-batch
        for m in range(nOfMinibatches):

            # Compute the starting index of this mini-batch
            startIdx = m * minibatchSize

            # Declare sampled inputs and outputs
            xSample = X[fullIdx[startIdx:startIdx + minibatchSize]]
            ySample = Y[fullIdx[startIdx:startIdx + minibatchSize]]

            # Run backprop, with an optimizer
            backProp(weights, xSample, ySample, learningRate, optimizer, mu)

        # Check cost and accuracy
        cost = nnCost(weights, X, Y)
        yes = evaluate(weights, X, Y)
        print("\nEpoch " + str(epoch + 1) + " of " + str(nEpochs) + " : " +
              str(yes) + " of " + str(len(Y)) + " = " + str(round(float(yes / len(Y)), 4)) +
              "; cost=" + str(cost), end='')

        if testY is not Y:
            testCost = nnCost(weights, testX, testY)
            testYes = evaluate(weights, testX, testY)
            print("  Val: " + str(testYes) + " of " + str(len(testY)) + " = " +
                str(round(float(testYes / len(testY)), 4)) + "; cost=" + str(testCost))

        # If decay type is 'step', when cost increases, revert back weights and
        # halve learning rate
        if decay is 'step':
            # If cost does not decrease
            if cost >= prevCost:
                # Revert weights back to those at the start of this epoch
                weights = [np.array(w) for w in oldWeights]

                # Recalculate prevCost
                cost = nnCost(weights, testX, testY)

                # Halve the learning rate
                learningRate = learningRate / 2.0

                # Revert iteration number
                epoch -= 1

                # Increment the count of halving learning rate by 1
                countLRHalf += 1

                print("Halving learning rate to: " +
                      str(learningRate) + ", count=" + str(countLRHalf))
            # If cost decreases, reset the count to 0
            else:
                countLRHalf = 0

        # If decay is 'time'
        if decay is 'time':
            learningRate *= np.exp(-decayRate)

        # If learningRate has been halved contiguously for too long, break
        if countLRHalf is 10:
            break

        # Set prevCost for next epoch
        prevCost = cost

        # Set oldWeights for next epoch
        oldWeights = [np.array(w) for w in weights]

        # Increase iteration number for epochs
        epoch += 1

    # If training was stopped because accuracy was not increasing
    if epoch < nEpochs:
        print("Training ended prematurely...")
    # If training ended in correct number of epochs
    else:
        print("Training complete.")

    # Printing training accuracy
    cost = nnCost(weights, X, Y)
    yes = evaluate(weights, X, Y)
    print("TRAINING ACCURACY, COST : " + str(yes) + " of " + str(len(Y)) +
          " = " + str(round(float(yes / len(Y)), 4)) + "; cost=" + str(cost))

    # Printing test accuracy
    if testY is not Y:
        testYes=evaluate(weights, testX, testY)
        print("TEST ACCURACY, COST : " + str(testYes) + " of " + str(len(testY)) +
        " = " + str(round(float(testYes / len(testY)), 4)) + "; cost=" + str(testCost))


# IMPLEMENTING BACK-PROPAGATION WITH LEARNING RATE, MOMENTUM, NAG, ADAGRAD
def backProp(weights, X, Y, learningRate, optimizer=None, mu=0.9):
    # Forward propagate to find outputs
    outputs=forwardProp(X, weights)

    # For the last layer, bpError = error = yPred - Y
    bpError=outputs[-1] - Y

    # Initialize velocity in the shape of weights for use with momentum and NAG
    v=[np.zeros(w.shape) for w in weights]
    prevV=[np.zeros(w.shape) for w in weights]

    # Initialize cache for use with Adagrad
    cache=[np.zeros(w.shape) for w in weights]

    # Back-propagating from the last layer to the first
    for l, w in enumerate(reversed(weights)):

        # Find yPred for this layer
        yPred=outputs[-l - 1]

        # Calculate delta for this layer using bpError from next layer
        delta=np.multiply(np.multiply(bpError, yPred), 1 - yPred)

        # Find input to the layer, by adding bias to the output of the previous layer
        # Take care, l goes from 0 to 1, while the weights are in reverse order
        if l == len(weights) - 1:  # If 1st layer has been reached
            xL=addBiasTerms(X)
        else:
            xL=addBiasTerms(outputs[-l - 2])

        # Calculate the gradient for this layer
        grad=np.dot(delta.T, xL) / len(Y)

        # Calculate bpError for previous layer to be back-propagated
        bpError=np.dot(delta, w)

        # Ignore bias term in bpError
        bpError=bpError[:, 1:]

        # CHANGE WEIGHTS of the current layer (W <- W + eta*deltaW)
        if optimizer is None:
            w += -learningRate * grad

        # Momentum
        if optimizer is 'momentum':
            v[-l - 1]=mu * v[-l - 1] - learningRate * grad
            w += v[-l - 1]

        # Nesterov Momentum
        if optimizer is 'nag':
            prevV[-l - 1]=np.array(v[-l - 1])  # back this up
            v[-l - 1]=mu * v[-l - 1] - learningRate * \
                grad  # velocity update stays the same
            # position update changes form
            w += -mu * prevV[-l - 1] + (1 + mu) * v[-l - 1]

        # Adagrad
        if optimizer is 'adagrad':
            cache[-l - 1] += grad**2
            w += - learningRate * grad / \
                (np.sqrt(cache[-l - 1]) + np.finfo(float).eps)

# Initialize network
layers=[2, 2, 1]
weights=initializeWeights(layers)

print("weights:")
for i in range(len(weights)):
    print(i + 1)
    print(weights[i].shape)
    print(weights[i])

# Declare input and desired output for AND gate
X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y=np.array([[0], [0], [0], [1]])

# Check current accuracy and cost
print("Cost: " + str(nnCost(weights, X, Y)))
yes=evaluate(weights, X, Y)
print("Accuracy: " + str(yes) + " of " + str(len(Y)) +
      " = " + str(round(float(yes / len(Y)), 4)))
print(forwardProp(X, weights)[-1])
