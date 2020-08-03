import numpy as np
import common, random
from collections import Counter

trainsize = 24000
testsize = 2000

word2index = common.get_word_index_imdb()
np.random.seed(1)

def prepare_data(posfilpath, negfilepath, size):
    posfiles = common.get_files(posfilpath)
    negfiles = common.get_files(negfilepath)
    sz = min(2*len(posfiles), 2*len(negfiles), size)

    x_train = list()
    x_concat = list()

    for i in range(int(size/2)):
        posline = common.get_content(posfilpath + posfiles[i])
        if (posline != ""):
            sent_pos = posline.lower().split(" ")
            sent_ls = list()
            for word in sent_pos:
                if word in word2index:
                    word_i = word2index[word]
                    sent_ls.append(word_i)
                    x_concat.append(word_i)
            x_train.append(sent_ls)
        
        negline = common.get_content(negfilepath + negfiles[i])

        if (negline != ""):
            sent_neg = negline.lower().split(" ")
            sent_ls = list()
            for word in sent_neg:
                if word in word2index:
                    word_i = word2index[word]
                    sent_ls.append(word_i)
                    x_concat.append(word_i)
            x_train.append(sent_ls)

    return (x_train, x_concat, sz)

(x_train, x_concat, sz) = prepare_data(common.POS_FILE_PATH, common.NEG_FILE_PATH, trainsize)
x_concat = np.array(x_concat)
print(len(x_train), len(x_concat), sz)

random.shuffle(x_train)
(alpha, iterations, hidden_size, window, negative) = (0.05, 2, 50, 2, 5)

w1 = (np.random.rand(len(word2index), hidden_size) - 0.5)*0.2
w2 = np.random.rand(len(word2index), hidden_size)*0

l2_target = np.zeros(negative + 1)
l2_target[0] = 1


def train2l(w1, w2):
    for rev_i, review in enumerate(x_train*iterations):
        for target_i in range(len(review)):
            target_samples = [review[target_i]] + list(x_concat[(np.random.rand(negative)*len(x_concat)).astype('int').tolist()])
            left_context = review[max(0, target_i -window):target_i]
            right_context = review[target_i+1 : min(len(review), target_i + window)]

            try :
                l1 = np.mean(w1[left_context + right_context], axis=0)
                l2 = common.sigmoid(l1.dot(w2[target_samples].T))

                d2 = l2 - l2_target
                d1 = d2.dot(w2[target_samples])

                w1[left_context + right_context] -= d1*alpha
                w2[target_samples] -= np.outer(d2, l1)*alpha
            except Exception as e:
                print ("in nlpblankNN.py : " , l1.dot(w2[target_samples].T), rev_i, target_i, e, target_samples)
                return (w1, w2)
            # finally :
            #     print (w2[target_samples].T, rev_i, target_i, target_samples)

            # if (rev_i == 0 and target_i == 3):
            #     print(l1.shape, l2.shape, w2[target_samples].shape, d1.shape, d2.shape)
            #     return
    return (w1, w2)
        

(nw1, nw2) = train2l(w1, w2)

#write the weights to file so that they can be reloaded
np.savetxt('nlpNNBlank_w1.txt', nw1, delimiter=",")
np.savetxt('nlpNNBlank_w2.txt', nw2, delimiter=",")

print(common.similar('king', nw1, nw2, word2index))