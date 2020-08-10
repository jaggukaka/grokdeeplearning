import numpy as np
import tensors as tns

# Simple nueral networks using Tensors
np.random.seed(0)

data = tns.Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True, id="data")
target = tns.Tensor(np.array([[0],[1],[0],[1]]), autograd=True, id="target")

w = list()
w.append(tns.Tensor(np.random.rand(2, 3), autograd=True, id="w0"))
w.append(tns.Tensor(np.random.rand(3, 1), autograd=True, id="w1"))

optim = tns.SGD(w, alpha = 0.1)

for i in range(10):

    pred = data.mm(w[0]).mm(w[1])
    k = pred - target
    loss = (k*k).sum(0)

    loss.backward(tns.Tensor(np.ones_like(loss.data)))
    optim.step()

    #print(loss)

#print('\n\n')

#Simple multi layered NN with tensor framework with Layer class
seq_layer = tns.Sequential([tns.Linear(2,3), tns.Tanh(), tns.Linear(3,1), tns.Sigmoid()])
criterion = tns.MSELoss()

optim = tns.SGD(parameters=seq_layer.get_parameters(), alpha = 1)

for i in range(10):

    pred = seq_layer.forward(data)
    loss = criterion.forward(pred, target)

    loss.backward(tns.Tensor(np.ones_like(loss.data)))
    optim.step()

    #print(loss)


#NN for bag of words or word embeddings

data = tns.Tensor(np.array([1,2,1,2]), autograd=True)

target = tns.Tensor(np.array([0,1,0,1]), autograd=True)

model = tns.Sequential([tns.Embedding(3,3), tns.Tanh(), tns.Linear(3,4)])
criterion = tns.CrossEntropyLoss()

optim = tns.SGD(parameters=model.get_parameters(), alpha=0.1)

for i in range(10):

    pred = model.forward(data)

    loss = criterion.forward(pred, target)

    loss.backward(tns.Tensor(np.ones_like(loss.data)))
    optim.step()

    #print(loss)


##RNN using word embeds and RNN layer

f = open('./qa.txt','r')
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n","").split(" ")[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)

tokens = new_tokens

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i
    
def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx

indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)

data = np.array(indices)

#print(tokens[:3])

embed = tns.Embedding(len(vocab), dim=16)
model = tns.RNN(n_input=16, n_hidden=16, n_output=(len(vocab)))

criterion = tns.CrossEntropyLoss()

optim = tns.SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)


for i in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):
        input = tns.Tensor(data[0:batch_size, t], autograd=True)
        #print (input, input.data.shape)
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

    target = tns.Tensor(data[0:batch_size, t+1], autograd=True)
    loss = criterion.forward(output, target)
    loss.backward()
    optim.step()

    total_loss += loss.data
    if(i % 200 == 0):
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print("Loss:",total_loss / (len(data)/batch_size),"% Correct:",p_correct)


batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)

for t in range(5):
    input = tns.Tensor(data[0:batch_size, t], autograd=True)
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = tns.Tensor(data[0:batch_size, t+1], autograd=True)
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context : " + ctx)
print("True : " + vocab[data[0:batch_size][0][-1]])
print("Pred : " + vocab[output.data.argmax()])