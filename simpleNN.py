import numpy as np
import tensors

ih_wgt = np.array([[0.5, 0.4, 0.7], [0.3, 0.3, -0.3], [-0.3, 0.4, 0.9]])
ho_wgt = np.array([[0.6, 0.3, 0.1], [0.3, 0.1, 0.8], [-0.9, 0.4, 0.1]])

weights = [ih_wgt, ho_wgt]

def neural_network(input, weights) :
    hid = input.dot(weights[0])
    pred = hid.dot(weights[1])
    return pred

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])
# pred = neural_network(input, weights)

# print (pred)

############## Gradient descent very simple #####################

input = 5
goal = 0.8
weight = 0.4
alpha = 0.01

def train(input, weight, goal, n):

    for _ in range(n):
        pred = input*weight
        delta = pred - goal
        error = delta**2 # can also be written as error = (0.5*weight - 0.8)**2 this is the crux of gradient descent

        weight -= delta*input*alpha
        print(weight, error)

#train(input, weight, goal, 20)

toes = [2.1, 2.6, 4.3, 1.5]
wlrec = [0.62, 0.71, 0.56, 0.29]
nfans = [0.55, 0.05, 0.55, 0.03]

winloss = np.array([1, 1, 1, 0])

input = np.array([[toes[0], wlrec[0], nfans[0]]])
goal = [winloss[0]]
weight = np.array([0.1, 0.2, -0.1])
alpha = 0.1

# gradient descent and stochastic gradient descent
def trainvector(input, weight, goal, n):

    for _ in range(n):
        for i in range(len(input)):
            pred = input[i].dot(weight)
            delta = pred - goal[i]
            error = delta**2
            #print (delta)
            weight -= alpha*delta*input[i]
            print (weight, error)
    return weight

def predvector(weight, k):

    for i in range(k):
        input = np.array([toes[i], wlrec[i], nfans[i]])
        pred = input.dot(weight)

        print (pred)


#weight = trainvector(input, weight, goal, 10) # gradient descent example (single sample)
#predvector(weight, 4)

############################### end of simple single input with different features and weights single output network ######################

############################### start of multiples samples of input and weights with multiple output ###############################


input = np.array([[2.1, 2.6, 4.3, 1.5], [0.62, 0.71, 0.56, 0.29], [0.55, 0.05, 0.55, 0.03]])
goal = winloss
weight = np.array([0.1, 0.2, -0.1])
alpha = 0.1

#weight = trainvector(input.transpose(), weight, goal, 10) # stochastic gradient descent multiple samples.

 




############################################ Neural network with multiple layers (2) ##################################

# l0 - input layer
# w1 - weights from l0 to l1
# l1 - intermediate layer
# w2 - weights from l1 to l2
# l2 - final prediction
# y  - actual result label
# d1 - delta between l0 and l1
# d2 - delta between l1 and l2
# input - all input data samples

np.random.seed(1)
def relu(x):
    return (x > 0)*x

def reluderiv(x):
    return x > 0

input = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]]) 

y = np.array([1, 1, 0, 0])
alpha = 0.2
hiddensize = 4

w1 = 2*np.random.random((3, hiddensize)) - 1 #weights of layer 1
w2 = 2*np.random.random((hiddensize, 1)) - 1 # weights of layer 2

def train2l(input, w1, w2, n):

    for k in range(n):
        l2_error = 0
        for i in range(len(input)):
            l0 = input[i:i+1]
            l1 = relu(l0.dot(w1))
            l2 = l1.dot(w2)
            d2 = l2 - y[i:i+1]
            l2_error += np.sum(d2**2)

            d1 = d2.dot(w2.T)*reluderiv(l1)
            
            w2 -= l1.T.dot(d2)*alpha
            w1 -= l0.T.dot(d1)*alpha

        if (k % 10 == 9):
            print (l2_error)


train2l(input, w1, w2, 60)




