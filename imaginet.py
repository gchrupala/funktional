from layer import *
import util
from util import CosineDistance
from util import autoassign, params, clipped_rectify

class Visual(Layer):
    """Encode sequence of (embedded) words into a visual vector."""

    def __init__(self, size_embed, size, size_out, depth, gru_activation=tanh):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth, activation=self.gru_activation)
        self.Project = Dense(self.size, self.size_out)
        self.params = params(self.Encode, self.Project)

    def __call__(self, inp):
        return self.Project(last(self.Encode(inp)))

class LM(Layer):
    """Predict next word in sequence of (embedded) words."""

    def __init__(self, size_embed, size, depth, gru_activation=tanh):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth, activation=self.gru_activation)
        self.Predict = Dense(self.size, self.size_embed)
        self.params = params(self.Encode, self.Predict)

    def __call__(self, inp, *_): # Ignores other inputs
        return self.Predict(self.Encode(inp))

class AE(Layer):
    """Autoencode a sequence of (embedded) words."""

    def __init__(self, size_embed, size, depth, gru_activation=tanh):
        autoassign(locals())
        encoder = lambda size_in, size: StackedGRUH0(size_embed, size, self.depth, activation=self.gru_activation)
        decoder = lambda size_in, size: StackedGRU(size_embed, size, self.depth, activation=self.gru_activation)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Predict   = Dense(size_in=self.size, size_out=self.size_embed)
        self.params    = params(self.Encdec, self.Predict)

    def __call__(self, inp, out_prev):
        return self.Predict(self.Encdec(inp, out_prev))


class Multitask(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth, textual, gru_activation=tanh):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  = Visual(self.size_embed, self.size, self.size_out, self.depth, gru_activation=self.gru_activation)
        self.Textual = textual(self.size_embed, self.size, self.depth, gru_activation=self.gru_activation)
        self.params  = params(self.Embed, self.Visual, self.Textual)

    def __call__(self, inp, *args):
        inp_e = self.Embed(inp)
        rest  = [ self.Embed(arg) for arg in args ]
        img   = self.Visual(inp_e)
        txt   = softmax3d(self.Embed.unembed(self.Textual(*rest)))
        return (img, txt)

def MultitaskLM(size_vocab, size_embed, size, size_out, depth, gru_activation=tanh):
    """Visual encoder combined with a language model."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, LM, gru_activation=gru_activation)

def MultitaskAE(size_vocab, size_embed, size, size_out, depth, gru_activation=tanh):
    """Visual encoder combined with a recurrent autoencoder."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, AE, gru_activation=gru_activation)

        
class Imaginet(object):
    """Trainable imaginet model."""

    def __init__(self, size_vocab, size_embed, size, size_out, depth, network, alpha=0.5, 
                 gru_activation=tanh, cost_visual=CosineDistance):
        autoassign(locals())
        self.network = network(self.size_vocab, self.size_embed, self.size, self.size_out, self.depth, 
                               gru_activation=self.gru_activation)
                               
        input         = T.imatrix()
        output_t_prev = T.imatrix()
        output_t      = T.imatrix()
        output_v      = T.fmatrix()
        OH = OneHot(size_in=self.size_vocab)
        output_t_oh   = OH(output_t)
        output_v_pred, output_t_pred = self.network(input, output_t_prev)
        cost_T = CrossEntropy(output_t_oh, output_t_pred)
        cost_V = self.cost_visual(output_v, output_v_pred)
        cost = self.alpha * cost_T + (1.0 - self.alpha) * cost_V
        self.updater = Adam()
        updates = self.updater.get_updates(self.network.params, cost)
        self.train = theano.function([input, output_v, output_t_prev, output_t ], 
                                      [cost, cost_T, cost_V], updates=updates)
        self.predict = theano.function([input, output_t_prev], [output_v_pred, output_t_pred])

        # Like train, but no updates
        self.loss = theano.function([input, output_v, output_t_prev, output_t ], cost)

