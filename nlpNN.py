import numpy as np
from os import walk

#Read the aclImdb data, prepare data for training and then train

trainsize = 24000
testsize = 2000
POS_FILE_PATH = './aclImdb/train/pos/'
NEG_FILE_PATH = './aclImdb/train/neg/'

TEST_POS = './aclImdb/test/pos/'
TEST_NEG = './aclImdb/test/neg/'

def get_files(path):
    f = []
    for (_,_,filenames) in walk(path):
        f.extend(filenames)
    return f

def get_content(path):
    with open(path, '+r') as f:
        line = f.readline()
        if line:
            return line
    return ""


word2index = {}
with open('./aclImdb/imdb.vocab', '+r') as f:
    i = 0
    while 1:
        line = f.readline()
        if not line:
            break
        word2index[line.strip()] = i
        i += 1

def prepare_data(posfilpath, negfilepath, size):
    posfiles = get_files(posfilpath)
    negfiles = get_files(negfilepath)
    sz = min(2*len(posfiles), 2*len(negfiles), size)

    x_train = list()
    y_train = np.zeros((trainsize, 1))

    for i in range(int(trainsize/2)):
        posline = get_content(posfilpath + posfiles[i])
        k = 2*i
        if (posline != ""):
            sent_pos = posline.split(" ")
            sent_ls = list()
            for word in sent_pos:
                if word in word2index:
                    sent_ls.append(word2index[word])
                    y_train[k][0] = 1
            x_train.append(list(set(sent_ls)))
        
        negline = get_content(negfilepath + negfiles[i])

        if (negline != ""):
            sent_neg = negline.split(" ")
            sent_ls = list()
            for word in sent_neg:
                if word in word2index:
                    sent_ls.append(word2index[word])
            x_train.append(list(set(sent_ls)))
    return (x_train, y_train, sz)
    
(x_train, y_train, trainsize) = prepare_data(POS_FILE_PATH, NEG_FILE_PATH, trainsize)
(x_test, y_test, testsize) = prepare_data(TEST_POS, TEST_NEG, testsize)

np.random.seed(1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)

(iterations, hidden_size, alpha) = (2, 100, 0.01)

w1 = 0.2*np.random.random((len(word2index), hidden_size)) - 0.1
w2 = 0.2*np.random.random((hidden_size, 1)) - 0.1

def train2l(input, w1, w2):
    for n in range(iterations):
        correct = 0
        for i in range(trainsize):
            y = y_train[i][0]
            l0 = input[i]
            l1 = sigmoid(np.sum(w1[l0], axis=0))
            l2 = sigmoid(l1.dot(w2))

            d2 = l2 - y
            d1 = d2.dot(w2.T)*sigmoid_deriv(l1)

            w1[l0] -= d1*alpha
            w2 -= np.outer(l1, d2)*alpha

            if (np.abs(d2) < 0.5):
                correct += 1
            
            # if (i == 0):
            #     print (l1.shape, l2.shape, d2.shape, d1.shape, w1[l0].shape)
            
        
        print ("iter = " + str(n), "Train accuracy = " + str(correct/trainsize))

    correct = 0
    #write down weights learned here into a file so that it can be re-used later
    for i in range(testsize):
        y = y_test[i][0]
        l0 = x_test[i]
        l1 = sigmoid(np.sum(w1[l0], axis = 0))
        l2 = sigmoid(l1.dot(w2))
        d2 = l2 -y

        if (np.abs(d2) < 0.5):
            correct += 1
        
    print ("Test accuracy = " + str(correct/testsize))

    return (w1, w2)


(nw1, nw2) = train2l(x_train, w1, w2)
print (nw1.shape, nw2.shape)

#write the weights to file so that they can be reloaded
np.savetxt('nlpNN_w1.txt', nw1, delimiter=",")
np.savetxt('nlpNN_w2.txt', nw2, delimiter=",")

            
    


