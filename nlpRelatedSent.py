import numpy as np
import random

def softmax(x):
    x = np.atleast_2d(x)
    temp = np.exp(x)
    return temp/np.sum(temp, axis=1, keepdims=True)

wordembed = {}
wordlist = ['yankees', 'bears', 'braves', 'red', 'socks', 'lose', 'defeat', 'beat', 'tie']
for word in wordlist:
    wordembed[word] = np.array([[0., 0., 0.]]) # here we fixing the size of hidden layer in the network to 3

'''
Idea is to predict every word in the sentence starting from the beginning of the sentence, so first write a simple algorithm to do it for a static sentence so that 
later is can be extended to a sentence of variable length

for convenience let us follow the nomenclature as below
w - word
s - sentence
n - hidden network size (e.g. each word will have a 1xn or nx1 vector)
k - vocabulary size (all words)
m - length of sentence (# of words in a sentence)
I - identity matrix (also called reccurent matrix when we apply the generic algorithm), dims of this would be
    nxn
d - decoder which will transform last layer (?) to output simply said, so dims of this would be nxk
'''

#FORWARD PROPOGATION

#decoder is something which you can think as a layer just before predicting the output, so dimensions of it should be nxk, in our
#example n = 3, k = len(wordlist)
decoder = (np.random.rand(3, len(wordlist)) - 0.5)*0.2
alpha = 0.01
identity = np.eye(3)

#Now write a simple Recurrent network step by step for the sentence "Red socks defeat ?", answer would be 'yankees' but it won't be part of reccurent network
#It would be part of prediction, after all we have to predict it isn't it?

layer_0 = wordembed['red']
layer_1 = layer_0.dot(identity) + wordembed['socks']
layer_2 = layer_1.dot(identity) + wordembed['defeat']

#if you observe layer_2 is nothing but sum of wordembeds for 'red', 'socks' and 'defeat', because the dot product with I would absolutely change nothing.

pred = softmax(layer_2.dot(decoder)) # this is our prediction, nothing different same as weight*input of a basic NN

#BACK PROPOGATION
y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]) #since our target output for the last word is 'yankees' and 'yankees' is the first word in the wordembed

pred_delta = pred - y
layer_2_delta = pred_delta.dot(decoder.T)
defeat_delta = layer_2_delta * 1
layer_1_delta = layer_2_delta.dot(identity.T)
socks_delta = layer_1_delta * 1
layer_0_delta = layer_1_delta.dot(identity.T)
red_delta = layer_0_delta

#update word embeddings
wordembed['red'] -= red_delta*alpha
wordembed['socks'] -= socks_delta*alpha
wordembed['defeat'] -= defeat_delta*alpha

identity -= np.outer(layer_0, layer_1_delta)*alpha
identity -= np.outer(layer_1, layer_2_delta)*alpha

decoder -= np.outer(layer_2, pred_delta)*alpha

'''
what we did was trying to predict the last word based on first 3 words and in backpropogation we are updating the weights accordingly as per how much we have missed the prediction
But this is not a sustainable model, we need to generalize this for any length of sentence, so what we would do is write the same algorithm in a different way
so that the algorithm tries to predict every next word given a single word, we would calculate error for each prediction and moved forward in forward propogation.
In back prop we will update weights based on each prediction on how wrong we were, this could be a tricky implementation!

'''

sentences = list()
with open('./qa.txt', 'r+') as f:
    sentences = list(map(lambda x: x.lower().replace('\n', '').split(' ')[1:], f.readlines()[:2000]))


print (sentences[:3])

vocab = set()
for sent in sentences:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
    
def sent2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

def softmax_n(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


np.random.seed(1)
embed_size = 10
n_words = len(word2index)
n_sentences = len(sentences)

decoder = (np.random.rand(embed_size, n_words) - 0.5)*0.1
alpha = 0.001
word_embeds = (np.random.rand(n_words, embed_size) - 0.5)*0.1
recurrent = np.eye(embed_size)
start = np.zeros(embed_size)

one_hot = np.eye(n_words)


def predict(sent):
    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0
    for k in range(len(sent)):
        layer = {}

        layer['pred'] = softmax_n(layers[-1]['hidden'].dot(decoder))
        loss += -np.log(layer['pred'][sent[k]])
        # if (np.isnan(loss)) :
        #     print (layer['pred'], sent[k], layers[-1]['hidden'], decoder, sent)
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + word_embeds[sent[k]]
        layers.append(layer)
    
    return layers, loss


def train():
    global word_embeds, recurrent, decoder, start
    for i in range(30000):
        sent_d = sentences[i%n_sentences][1:]
        sent = sent2indices(sent_d)
        
        layers, loss = predict(sent)
        n_layers = len(layers)
        n_sent = float(len(sent))
        for k in reversed(range(n_layers)):
            layer = layers[k]
            target = sent[k - 1]

            if (k > 0):
                layer['pred_delta'] = layer['pred'] - one_hot[target]
                hidden_delta = layer['pred_delta'].dot(decoder.T)

                if (k == n_layers -1): # last layer
                    layer['hidden_delta'] = hidden_delta
                else:
                    layer['hidden_delta'] = hidden_delta + layers[k+1]['hidden_delta'].dot(recurrent.T)
            else:
               layer['hidden_delta'] =  layers[k+1]['hidden_delta'].dot(recurrent.T)


        # we should effectively update start, decoder and recurrent, because these are the 
        alpha_s = alpha/n_sent
        start -= layers[0]['hidden_delta']*alpha_s
        for k, layer in enumerate(layers[1:]):
            decoder -= np.outer(layers[k]['hidden'], layer['pred_delta'])*alpha_s
            word_i = sent[k]
            #print (word_embeds[word_i])
            word_embeds[word_i] -= layers[k]['hidden_delta']*alpha_s
            
            #print (word_embeds[word_i], layer['hidden_delta'][0])
            recurrent -= np.outer(layers[k]['hidden'], layer['hidden_delta'])*alpha_s
        
        
        # if (np.isnan(perplex)) :
        #     print (n_sent, loss, sent, sent_d,i)
        #     exit(1)
        
        if (i % 1000 == 0):
            perplex = np.exp(loss/n_sent)
            print ("Perplexity : " + str(perplex))



train()


sent_d = sentences[4]
print (sent_d)
sent = sent2indices(sent_d)
layers, _ = predict(sent)

for k, layer in enumerate(layers[1:-1]):
    input = sent_d[k]
    true = sent_d[k+1]

    pred = vocab[layer['pred'].argmax()]
    print (input, true, pred)