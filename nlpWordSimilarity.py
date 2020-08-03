import numpy as np
import common, math, _thread
from collections import Counter

'''
First load the weights which were trained in the nlpNN.py script using Imdb data set, before running this script make sure you run the 
nlpNN.py script
'''

trainsize = 24000

#Following are the weights learned from the script nlpNN.py on Imdb data set
# w1 = np.loadtxt('nlpNN_w1.txt', delimiter=",")
# w2 = np.loadtxt('nlpNN_w2.txt', delimiter=",")

#print (w1.shape, w2.shape)

word2index = common.get_word_index_imdb()

def similar_nlpNN(target = 'horrible'):
    return common.similar(target, w1, w2, word2index)

#print("The similarity when compared with simple nlp NN with word embeddings : ", similar_nlpNN('good'))

w1 = np.loadtxt('nlpNNBlank_w1.txt', delimiter=",")
w2 = np.loadtxt('nlpNNBlank_w2.txt', delimiter=",")
    

#Similar reviews implementation

def makesentvec(sent):
    try :
        word_indices = list(map(lambda x: word2index[x], filter(lambda x: x in word2index, sent)))
        return np.mean(normw[word_indices], axis=0)
    except Exception as e:
        print (sent, word_indices, e)
        exit(1)

def prepare_data(posfilpath, negfilepath, size):
    posfiles = common.get_files(posfilpath)
    negfiles = common.get_files(negfilepath)

    reviews = list()
    raw_sent = list()
    for i in range(int(size/2)):
        posline = common.get_content(posfilpath + posfiles[i])
        if (posline != ""):
            posline = posline.lower()
            sent = posline.split(" ")
            reviews.append(makesentvec(sent))
            raw_sent.append(posline)
        
        negline = common.get_content(negfilepath + negfiles[i])

        if (negline != ""):
            negline = negline.lower()
            sent = negline.split(" ")
            reviews.append(makesentvec(sent))
            raw_sent.append(negline)

    return (reviews, raw_sent)
    
normw = common.normalize(w1)
(sent_vectors, raw_sent) = prepare_data(common.POS_FILE_PATH, common.NEG_FILE_PATH, trainsize) 

sent_vectors = np.array(sent_vectors)


def most_similar_review(review):
    review_vec = makesentvec(review)
    scores = Counter()
    k = sent_vectors.dot(review_vec)
    for i, val in enumerate(k):
        scores[i] = val
    most_similar = list()

    #print(k.shape, sent_vectors.shape, scores.most_common(10))
    for i, _ in scores.most_common(5):
        most_similar.append(raw_sent[i])
    return most_similar


def takeInput():
    while 1:
        value = input("Please enter messageType:\n")
        value = value.split(" ")
        if value[0].isdigit():
            val = int(value[0])
            if val == 0:
                print("The similarity when compared with nlp NN with fill in the blank type loss function : ", similar_nlpNN(value[1]))                
            elif val == 2:
                [print(i, x) for i, x in enumerate(most_similar_review(value[1:]))]
            else :
                print(common.analogy(value[1:val], value[val + 1: len(value)], w1, w2, word2index))


print ("Program is taking input, proceed..")
takeInput()