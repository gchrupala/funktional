from layer import *
import util
from util import autoassign, params

class Visual(Layer):
    """Encode sequence of (embedded) words into a visual vector."""

    def __init__(self, size_embed, size, size_out, depth):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth)
        self.Project = Dense(self.size, self.size_out)
        self.params = params(self.Encode, self.Project)

    def __call__(self, inp):
        return self.Project(last(self.Encode(inp)))

class LM(Layer):
    """Predict next word in sequence of (embedded) words."""

    def __init__(self, size_embed, size, depth):
        autoassign(locals())
        self.Encode  = StackedGRUH0(self.size_embed, self.size, self.depth)
        self.Predict = Dense(self.size, self.size_embed)
        self.params = params(self.Encode, self.Predict)

    def __call__(self, inp):
        return self.Predict(self.Encode(inp))

class AE(Layer):
    """Autoencode a sequence of (embedded) words."""

    def __init__(self, size_embed, size, depth):
        autoassign(locals())
        encoder = lambda size_in, size: StackedGRUH0(size_embed, size, self.depth)
        decoder = lambda size_in, size: StackedGRU(size_embed, size, self.depth)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Predict   = Dense(size_in=self.size, size_out=self.size_embed)
        self.params    = params(self.Encdec, self.Predict)

    def __call__(self, inp, out_prev):
        return self.Predict(self.Encdec(inp, out_prev))


class Multitask(Layer):
    """Visual encoder combined with a textual task."""
    
    def __init__(self, size_vocab, size_embed, size, size_out, depth, textual):
        autoassign(locals())
        self.Embed   =  Embedding(self.size_vocab, self.size_embed)
        self.Visual  = Visual(self.size_embed, self.size, self.size_out, self.depth)
        self.Textual = textual(self.size_embed, self.size, self.depth)
        self.params  = params(self.Embed, self.Visual, self.Textual)

    def __call__(self, inp, *args):
        inp_e = self.Embed(inp)
        rest  = [ self.Embed(arg) for arg in args ]
        img   = self.Visual(inp_e)
        txt   = sofmax3d(self.Embed.unembed(self.Textual(*rest)))
        return (img, txt)

def MultitaskLM(size_vocab, size_embed, size, size_out, depth):
    """Visual encoder combined with a language model."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, LM)

def MultitaskAE(size_vocab, size_embed, size, size_out, depth):
    """Visual encoder combined with a recurrent autoencoder."""
    return Multitask(size_vocab, size_embed, size, size_out, depth, AE)

        
