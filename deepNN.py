import sys, numpy as np

trainsize = 1000
testsize = 500

x_train = np.loadtxt("./mnist_train.csv", delimiter=",", max_rows=trainsize)
x_test = np.loadtxt("./mnist_test.csv", delimiter=",", max_rows=testsize)
print (x_train.shape, x_test.shape)

y_train = x_train[:, :1]
y_test = x_test[:, :1]

x_train = x_train[:, 1:]/255
x_test = x_test[:, 1:]/255

feat_n = x_train.shape[1]

def tanh(x):
    return np.tanh(x)

def tanhderiv(x):
    return 1 - x**2

def softmax(x):
    k = np.exp(x)
    return k/np.sum(k, axis= 1, keepdims= True)

def hotones(x):
    hot_ones = np.zeros((len(y_train), 10))
    for l, k in enumerate(x):
        k = k[0].astype(np.int)
        hot_ones[l][k] = 1
    return hot_ones

def pred(l0, w1, w2):
    l1 = tanh(l0.dot(w1))
    l2 = softmax(l1.dot(w2))
    return l2


y_train = hotones(y_train)
y_test = hotones(y_test)

alpha, hidden_size, n = (2, 100, 300)

np.random.seed(1)
w1 = 0.2*np.random.random((feat_n, hidden_size)) - 0.1
w2 = 0.02*np.random.random((hidden_size, 10)) - 0.01


def train2l(input, w1, w2, n, batchsize):
    for k in range(n):
        correct_cnt = 0
        test_correct_cnt = 0
        for i in range(int(trainsize/batchsize)):
            bstart = i*batchsize
            bend = min(batchsize + bstart, trainsize)
            bsize = bend - bstart
            l0 = input[bstart:bend]
            l1 = tanh(l0.dot(w1))
            #print(l1.shape, l0.shape)
            dropout_mask = np.random.randint(2, size=l1.shape)
            l1 *= dropout_mask
            l2 = softmax(l1.dot(w2))

            for j in range(bsize):
                correct_cnt += int(np.argmax(l2[j:j+1]) == np.argmax([y_train[bstart+j:bstart+j+1]]))

            d2 = (l2 - y_train[bstart:bend])/(bsize)
            d1 = (d2.dot(w2.T)*tanhderiv(l1))*dropout_mask

            w1 -= l0.T.dot(d1)*alpha
            w2 -= l1.T.dot(d2)*alpha

        for m in range(testsize):
            l2 = pred(x_test[m:m+1], w1, w2)
            test_correct_cnt += int(np.argmax(l2) == np.argmax(y_test[m:m+1]))
        
        if (k % 10 == 0):
            print(k, "Test acc = " + str(test_correct_cnt/testsize), " Train acc = " + str(correct_cnt/trainsize) )

        

train2l(x_train, w1, w2, n, 100)



