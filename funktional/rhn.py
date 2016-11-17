# Based on code in
# https://github.com/julian121266/RecurrentHighwayNetworks/blob/master/theano_rhn.py
# by The Swiss AI lab IDSIA.
import numpy as np
import theano.tensor as tt
import context
from funktional.layer import Layer

# class Linear(Layer):
#     def __init__(self, in_size, out_size, bias, bias_init=None, init_scale=0.04):
#         autoassign(locals())
#         assert bias == (bias_init is not None)
#         self.w = self.make_param((in_size, out_size), 'uniform')
#         self.b = self.make_param((out_size,), bias_init)
    
#     def __call__(self, x):
#         y = tt.dot(x, self.w)
#         if self.bias:
#             y += self.b
#         return y

#     def params(self):
#         return [self.w, self.b]
    
#     def make_param(self, shape, init_scheme):
#         """Create Theano shared variables, which are used as trainable model parameters."""
#         if isinstance(init_scheme, numbers.Number):
#             init_value = np.full(shape, init_scheme, floatX)
#         elif init_scheme == 'uniform':
#             init_value = np.uniform(low=-self.init_scale, high=self.init_scale, size=shape).astype(floatX)
#         else:
#             raise AssertionError('unsupported init_scheme')
#         p = theano.shared(init_value)
#         return p

    
class RHN(Layer):
    """Recurrent Highway Network. Based on
    https://arxiv.org/abs/1607.03474 and
    https://github.com/julian121266/RecurrentHighwayNetworks.

    """
    def __init__(self, size_in, size, recur_depth=1, drop_i=0.75 , drop_s=0.25,
                 init_T_bias=-2.0, init_H_bias='uniform', tied_noise=True):
        autoassign(locals())
        self._theano_rng = RandomStreams(config.seed // 2 + 321)
        self._np_rng = np.random.RandomState(config.seed // 2 + 123)
        self._params = []

    def apply_dropout(self, x, noise):
        return ifelse(context.training, noise * x, x)

    def get_dropout_noise(self, shape, dropout_p):
        keep_p = 1 - dropout_p
        noise = cast_floatX(1. / keep_p) * self._theano_rng.binomial(size=shape, p=keep_p, n=1, dtype=floatX)
        return noise
    
    def params(self):
        return self._params

    def make_param(self, shape, init_scheme):
        """Create Theano shared variables, which are used as trainable model parameters."""
        if isinstance(init_scheme, numbers.Number):
            init_value = np.full(shape, init_scheme, floatX)
        elif init_scheme == 'uniform':
            init_value = self._np_rng.uniform(low=-self._init_scale, high=self._init_scale, size=shape).astype(floatX)
        else:
            raise AssertionError('unsupported init_scheme')
        p = theano.shared(init_value)
        self._params.append(p)
        return p

    def linear(self, x, in_size, out_size, bias, bias_init=None):
        assert bias == (bias_init is not None)
        w = self.make_param((in_size, out_size), 'uniform')
        y = tt.dot(x, w)
        if bias:
            b = self.make_param((out_size,), bias_init)
            y += b
        return y
   
    def step(self, i_for_H_t, i_for_T_t, h_tm1, noise_s):
        tanh, sigm = T.tanh, T.nnet.sigmoid
        noise_s_for_H = noise_s if self.tied_noise else noise_s[0]
        noise_s_for_T = noise_s if self.tied_noise else noise_s[1]
        
        s_lm1 = h_tm1
        for l in range(depth):
            s_lm1_for_H = self.apply_dropout(s_lm1, noise_s_for_H)
            s_lm1_for_T = self.apply_dropout(s_lm1, noise_s_for_T)
            if l == 0:
                # On the first micro-timestep of each timestep we already have bias
                # terms summed into i_for_H_t and into i_for_T_t.
                H = tanh(i_for_H_t + self.linear(s_lm1_for_H, in_size=size, out_size=hidden_size, bias=False))
                T = sigm(i_for_T_t + self.linear(s_lm1_for_T, in_size=hidden_size, out_size=hidden_size, bias=False))
            else:
                H = tanh(self.linear(s_lm1_for_H, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_H_bias))
                T = sigm(self.linear(s_lm1_for_T, in_size=hidden_size, out_size=hidden_size, bias=True, bias_init=init_T_bias))
            s_l = (H - s_lm1) * T + s_lm1
            s_lm1 = s_l

        y_t = s_l
        return y_t

    def __call__(self, h0, seq, repeat_h0=1):
        X = seq.dimshuffle((1,0,2))
        H0 = T.repeat(h0, X.shape[1], axis=0) if repeat_h0 else h0
        out, _ = theano.scan(self.step,
                             sequences=[i_for_H, i_for_T],
                             outputs_info=[H0],
                             non_sequences = [noise_s])
        return out.dimshuffle((1, 0, 2))

