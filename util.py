def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

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

def CategoricalCrossEntropySwapped(y_true, y_pred):
    return T.nnet.categorical_crossentropy(T.clip(y_pred, 1e-7, 1.0-1e-7), y_true).mean()

def MeanSquaredError(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()
