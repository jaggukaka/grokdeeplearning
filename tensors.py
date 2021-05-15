import numpy as np

class Tensor (object) :

    def __init__(self, data, create_op=None, creators=None, id=None, autograd=False):
        self.data = np.array(data)
        self.create_op = create_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        if(id is None):
            self.id = str(np.random.randint(0,100000))
        else:
            self.id = str(id)
        self.children = {}
        if (self.creators):
            for c in self.creators: 
                if (self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_done(self):
        for _, cnt in self.children.items():
            if (cnt != 0):
                return False
        return True

    def __add__(self, other):
        return Tensor(self.data + other.data, create_op="add", creators=[self, other], 
                id=self.id + '_' + other.id, autograd=(self.autograd and other.autograd))
    
    def __neg__(self):
        return Tensor(self.data*-1, create_op="neg", creators=[self], id='-'+self.id, autograd=self.autograd)

    def __sub__(self, other):
        return Tensor(self.data - other.data, create_op="sub", creators=[self, other], id=self.id + '_' + other.id, autograd=(self.autograd and other.autograd))

    def __mul__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data * other.data, create_op="mul", creators=[self, other], id=self.id + '_' + other.id, autograd=True)
        else:
            return Tensor(self.data * other.data)
    
    def sum(self, dim):
        return Tensor(self.data.sum(dim), create_op="sum_" + str(dim), creators=[self], id='+'+self.id, autograd=self.autograd)
    
    def expand(self, dim, copies):
        s = self.data.shape
        k = len(s)
        trans_vec = list(range(0, k))
        trans_vec.insert(dim, k)
        new_data = self.data.repeat(copies).reshape(list(s) + [copies]).transpose(trans_vec)
        return Tensor(new_data, create_op="expand_" + str(dim), creators=[self], id='ex_'+self.id, autograd=self.autograd)
    
    def mm(self, other):
        return Tensor(self.data.dot(other.data), create_op="mm", creators=[self, other], id=self.id + "_" + other.id, autograd=(self.autograd and other.autograd))

    def transpose(self):
        return Tensor(self.data.transpose(), create_op="transpose", creators=[self], id='T_'+self.id, autograd=self.autograd)

    
    def sigmoid(self):
        new = 1/(1 + np.exp(-self.data))
        if (self.autograd):
            return Tensor(
                new, create_op="sigmoid", creators=[self], id='sigm_' + self.id, autograd=True
            )
        return Tensor(new)
    
    def tanh(self):
        new = np.tanh(self.data)
        if (self.autograd):
            return Tensor(
                new, create_op="tanh", creators=[self], id='tanh_' + self.id, autograd=True
            )
        return Tensor(new)

    def index_select(self, indices):

        if(self.autograd):
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         create_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    
    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        #print (self.data.shape, temp.shape, t.shape, p.shape, target_dist.shape, loss.shape, target_indices.data.shape, softmax_output.shape)
    
        if(self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         create_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out


        return Tensor(loss)
        
    
    def backward(self, grad=None, grad_origin=None ):
        
        if (self.autograd):
            if(grad is None):
                grad = Tensor(np.ones_like(self.data))
            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if (not self.grad):
                self.grad = grad
            else:
                self.grad += grad
            
            assert grad.autograd == False

            if (self.creators and (self.all_children_done() or grad_origin is None)):
                c0 = self.creators[0]
                if (len(self.creators) > 1):
                    c1 = self.creators[1]
                if self.create_op == "add":
                    c0.backward(self.grad, self)
                    c1.backward(self.grad, self)
                elif self.create_op == "neg":
                    c0.backward(self.grad.__neg__())
                elif self.create_op == "sub":
                    c0.backward(self.grad, self)
                    c1.backward(self.grad.__neg__(), self)
                elif self.create_op == "mul":
                    new = self.grad*c1
                    c0.backward(new, self)
                    new = self.grad*c0
                    c1.backward(new, self)
                elif self.create_op == "mm":
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
                elif "expand" in self.create_op:
                    dim = int(self.create_op.split('_')[1])
                    c0.backward(self.grad.sum(dim))
                elif "sum" in self.create_op:
                    dim = int(self.create_op.split('_')[1])
                    c0.backward(self.grad.expand(dim, c0.data.shape[dim]))
                elif self.create_op == "transpose":
                    c0.backward(self.grad.transpose())
                elif self.create_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    c0.backward(self.grad*(self*(ones - self)))
                elif self.create_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    c0.backward(self.grad*(ones - self*self))
                elif(self.create_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

                elif(self.create_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))
                    #print (grad.data.shape, new_grad.shape, self.index_select_indices.data, indices_, grad_.shape)



    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class SGD(object):

    def __init__(self, parameters, alpha=0.01):
        self.parameters = parameters
        self.alpha = alpha

    def zeros(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data*self.alpha

            if (zero):
                p.grad.data *=0


class Layer(object):

    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Linear(Layer):

    def __init__(self, n_inp, n_out):
        super().__init__()
        self.w = Tensor(np.random.randn(n_inp, n_out) * np.sqrt(2.0/(n_inp)), autograd=True)
        self.bias = Tensor(np.zeros(n_out), autograd=True)
        self.parameters.append(self.w)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.w) + self.bias.expand(0, len(input.data))


class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params

class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        k = pred - target
        return (k*k).sum(0)


class Tanh(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()

class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


class Embedding(Layer):
    
    def __init__(self, vocab_size, dim):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        
        # this random initialiation style is just a convention from word2vec
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)
        
        self.parameters.append(self.weight)
    
    def forward(self, input):
        return self.weight.index_select(input)


class RNN(Layer):

    def __init__(self, n_input, n_hidden, n_output, activation="sigmoid"):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not defined!")

        self.w_ih = Linear(n_input, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        #print( from_prev_hidden.data.shape, input.data.shape)
        combined = self.w_ih.forward(input) + from_prev_hidden
        #print(combined, combined.data.shape)
        new_hidden = self.activation.forward(combined)
        #print(new_hidden)
        output = self.w_ho.forward(new_hidden)
        #print(output)
        return output, new_hidden
    
    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)





# t5 = Tensor([2,2,2,2,2], id="t5", autograd=True)
# t6 = Tensor([5,4,3,2,1], id="t6", autograd=True)
# t7 = Tensor([1,2,3,4,5], id="t7", autograd=True)
# t11 = Tensor([2,2,2,2,2], id="t11", autograd=True)

# t10 = t11 + t7
# t8 = t6 + t7
# t4 = t6 + t5
# t9 = t10 + t8
# k = -t4
# t3 = k + t8
# t2 = t5 + k
# t1 = t2 + t3
# t0 = t1 - t9


# t0.backward(Tensor(np.array([1,1,1,1,1]),  id="t0_grad"))
# print (t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, t6.grad, t7.grad, t8.grad, t9.grad, t10.grad, t11.grad, k.grad)

# z = Tensor(np.array([[1,2,3],[4,5,6]]), id="z", autograd=True)
# z1 = z.expand(0, 3)
# z2 = z1.expand(2, 4)
# print (z.data.shape, z,  z1.data.shape, z1, z2.data.shape, z2)