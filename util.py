# Code adapted from https://github.com/IndicoDataSolutions/Passage
# Copyright (c) 2015 IndicoDataSolutions

import theano
import theano.tensor as T
import numpy as np

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])

def tanh(x):
	return T.tanh(x)

def steeper_sigmoid(x):
	return 1./(1. + T.exp(-3.75 * x))

def softmax3d(inp): 
    x = inp.reshape((inp.shape[0]*inp.shape[1],inp.shape[2]))
    e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
    result = e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
    return result.reshape(inp.shape)

def CrossEntropy(y_true, y_pred):
    return T.nnet.categorical_crossentropy(T.clip(y_pred, 1e-7, 1.0-1e-7), y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

class Adam(object):
    """Adam: a Method for Stochastic Optimization, Kingma and Ba. http://arxiv.org/abs/1412.6980."""

    def __init__(self, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(floatX(0.))
        i_t = i + 1.
        fix1 = 1. - self.b1**(i_t)
        fix2 = 1. - self.b2**(i_t)
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates
