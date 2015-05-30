# A simple encoder-decoder example with funktional

import theano
import numpy
import json
import itertools
from layer import *

class EncoderDecoder(Layer):
    """A simple encoder-decoder net with shared input and output vocabulary."""
    def __init__(self, size_vocab, size, depth):
        self.size_vocab  = size_vocab
        self.size     = size
        self.depth    = depth
        self.OH       = OneHot(self.size_vocab)
        encoder = lambda *args: StackedGRU(*args, self.depth=3)
        decoder = lambda *args: StackedGRU(*args, self.depth=3)
        self.Encdec   = EncoderDecoderGRU(self.size_vocab, self.size, self.size_vocab, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Out      = Dense(size_in=self.size, size_out=self.size_vocab)
        self.params   = self.Encdec.params + self.Out.params
        
    def __call__(self, inp, out_prev):
        return softmax3d(self.Out(self.Encdec(self.OH(inp), self.OH(out_prev))))

class Model(object):
    """Trainable encoder-decoder model."""
    def __init__(self, size_vocab, size):
        self.size = size
        self.size_vocab = size_vocab
        self.network = EncoderDecoder(self.size_vocab, self.size)
        self.input       = T.imatrix()
        self.output_prev = T.imatrix()
        self.output      = T.imatrix()
        self.output_oh   = self.network.OH(self.output)
        self.output_pred = self.network(self.input, self.output_prev)
        self.cost = CrossEntropy(self.output_oh, self.output_pred)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.network.params, self.cost)
        self.train = theano.function([self.input, self.output_prev, self.output ], 
                                      self.cost, updates=self.updates)
        self.predict = theano.function([self.input, self.output_prev], self.output_pred)

def paraphrases(data, split='train'):
    """Yields pairs of sentences describing the same image from data."""
    for image in data['images']:
        if image['split'] == split:
            sentences = [ s['raw'] for s in image['sentences'] ]
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        yield (sentences[j], sentences[i])

def sentences(data, split='train'):
    """Yields sentences from data."""
    for image in data['images']:
        if image['split'] == split:
            for sentence in image['sentences']:
                yield sentence['raw']
        
def pad(xss, padding):
    max_len = max((len(xs) for xs in xss))
    def pad_one(xs):
        return [ padding for _ in range(0,(max_len-len(xs))) ] + xs
    return [ pad_one(xs) for xs in xss ]

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF G
    args = [iter(iterable)] * n
    return itertools.izip(*args)

def to_bytes(str):
    return [ ord(c) for c in str.encode('utf-8') ]
    
def from_bytes(bs):
    return ''.join( [ chr(b) for b in bs]).decode('utf-8')
 
def main():
    data = json.load(open('/home/gchrupala/repos/neuraltalk/data/flickr30k/dataset.json'))
    mb_size = 128
    model = Model(size_vocab=256, size=512)
    for epoch in range(1,6):
        for _j, item in enumerate(grouper(sentences(data), 128)):
            j = _j + 1
            mb = numpy.array(pad([ [ord(' ')]+to_bytes(s) for s in item], ord(' ')), dtype='int32')
            inp = mb[:,1:]
            out = mb[:,1:]
            out_prev = mb[:,0:-1]
            print epoch, j, model.train(inp, out_prev, out)
            if j % 50 == 0:
                pred = model.predict(inp, out_prev)
                for i in range(len(pred)):
                    orig = repr(from_bytes(inp[i]))
                    res = repr(''.join([ chr(b) for b in numpy.argmax(pred, axis=2)[i] ]))
                    print len(orig), orig
                    print len(res), res
    
if __name__ == '__main__':
    main()
