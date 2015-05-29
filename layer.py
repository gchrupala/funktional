import theano
import theano.tensor as T
from util import *

class Layer(object):

    def __init__(self):
        self.params = []

    def __call__(self, inp):
        raise NotImplementedError

    def compose(self, l2):
        l = Layer()
        l.__call__ = lambda inp: self(l2(inp))
        l.params = self.params + l2.params
        return l


class Embedding(Layer):
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.E = uniform((size_in, size_out))
        self.params = [self.E]

    def __call__(self, inp):
        return self.E[inp]

def theano_one_hot(idx, n):
    z = T.zeros((idx.shape[0], n))
    one_hot = T.set_subtensor(z[T.arange(idx.shape[0]), idx], 1)
    return one_hot        

class OneHot(Layer):

    def __init__(self, size_in):
        self.size_in = size_in
        self.params = []

    def __call__(self, inp):
        return theano_one_hot(inp.flatten(), self.size_in).reshape((inp.shape[0], inp.shape[1], self.size_in))
        

class Dense(Layer):
    
    def __init__(self, size_in, size_out):
        self.size_in = size_in
        self.size_out = size_out
        self.w = orthogonal((self.size_in, self.size_out))
        self.b = shared0s((self.size_out))
        self.params = [self.w, self.b]

    def __call__(self, inp):
        return T.dot(inp, self.w) + self.b

class GRU(object):

    def __init__(self, size_in, size):
        self.size_in = size_in
        self.size = size
        self.activation = tanh
        self.gate_activation = steeper_sigmoid
        self.init = orthogonal
        self.size = size

        self.w_z = self.init((self.size_in, self.size))
        self.w_r = self.init((self.size_in, self.size))

        self.u_z = self.init((self.size, self.size))
        self.u_r = self.init((self.size, self.size))

        self.b_z = shared0s((self.size))
        self.b_r = shared0s((self.size))

        self.w_h = self.init((self.size_in, self.size)) 
        self.u_h = self.init((self.size, self.size))
        self.b_h = shared0s((self.size))   

        self.params = [self.w_z, self.w_r, self.w_h, self.u_z, self.u_r, self.u_h, self.b_z, self.b_r, self.b_h]

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        return h_t

    def __call__(self, seq, h0, repeat_h0=0):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        out, _ = theano.scan(self.step, 
            sequences=[x_z, x_r, x_h], 
                             outputs_info=[H0], 
            non_sequences=[self.u_z, self.u_r, self.u_h]
        )
        return out.dimshuffle((1,0,2))
        
class Zeros(Layer):
    
    def __init__(self, size):
        self.size  = size
        self.zeros = theano.shared(numpy.asarray(numpy.zeros((1,self.size)), dtype=theano.config.floatX))
        self.params = [self.zeros]
    
    def __call__(self):
        return self.zeros

    
class EncoderDecoder(object):

    def __init__(self, embedding_size, size, vocab_size):
        self.embedding_size = embedding_size 
        self.size = size
        self.vocab_size = vocab_size
        self.Embed = Embedding(self.vocab_size, self.embedding_size)
        self.Encode = GRU(size_in=self.embedding_size, size=self.size)
        self.Decode = GRU(size_in=self.embedding_size, size=self.size)
        self.H0  = Zeros(size=self.size)
        self.Out = Dense(size_in=self.size, size_out=self.vocab_size) 
        self.params = sum([ l.params for l in [Embed, H0, Encode, Decode, Out] ], [])
        
    def __call__(self, inp, out_prev):
        last = lambda x: x.dimshuffle((1,0,2))[-1]
        return softmax3d(self.Out(self.Decode(self.Embed(out_prev), 
                                              last(self.Encode(self.Embed(inp), 
                                                               self.H0(), repeat_h0=1)))))
