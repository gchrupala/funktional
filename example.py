# A simple encoder-decoder example with funktional
from __future__ import division 
import theano
import numpy
import json
import random
import itertools
from layer import *
from passage.preprocessing import Tokenizer
import cPickle as pickle

class EncoderDecoder(Layer):
    """A simple encoder-decoder net with shared input and output vocabulary."""
    def __init__(self, size_vocab, size, depth):
        self.size_vocab  = size_vocab
        self.size     = size
        self.depth    = depth
        self.Embed    = Embedding(self.size_vocab, self.size)
        encoder = lambda size_in, size: StackedGRUH0(size_in, size, self.depth)
        decoder = lambda size_in, size: StackedGRU(size_in, size, self.depth)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Out      = Dense(size_in=self.size, size_out=self.size)
        self.params   = self.Embed.params + self.Encdec.params + self.Out.params
        
    def __call__(self, inp, out_prev):
        return softmax3d(self.Embed.debed(self.Out(self.Encdec(self.Embed(inp), self.Embed(out_prev)))))

class Model(object):
    """Trainable encoder-decoder model."""
    def __init__(self, size_vocab, size, depth):
        self.size = size
        self.size_vocab = size_vocab
        self.depth = depth
        self.network = EncoderDecoder(self.size_vocab, self.size, self.depth)
        self.input       = T.imatrix()
        self.output_prev = T.imatrix()
        self.output      = T.imatrix()
        OH = OneHot(size_in=self.size_vocab)
        self.output_oh   = OH(self.output)
        self.output_pred = self.network(self.input, self.output_prev)
        self.cost = CrossEntropy(self.output_oh, self.output_pred)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.network.params, self.cost)
        self.train = theano.function([self.input, self.output_prev, self.output ], 
                                      self.cost, updates=self.updates)
        self.predict = theano.function([self.input, self.output_prev], self.output_pred)
    
def sentences(data, split='train'):
    """Yields sentences from data."""
    for image in data:
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

def main():
    data = json.load(open('/home/gchrupala/repos/neuraltalk/data/coco/dataset.json'))['images']
    random.shuffle(data)
    tokenizer = Tokenizer(min_df=10)
    sents = list(sentences(data))
    tokenized = tokenizer.fit_transform(sents)
    PAD = tokenizer.encoder['PAD']
    END = tokenizer.encoder['END']
    mb_size = 128
    print tokenizer.n_features
    model = Model(size_vocab=tokenizer.n_features, size=512, depth=2)
    for epoch in range(1,11):
        costs = 0 ; N = 0
        for _j, item in enumerate(grouper(tokenized, 128)):
            j = _j + 1
            mb = numpy.array(pad([[PAD]+s+[END] for s in item], PAD), dtype='int32')
            inp = mb[:,1:]
            out = mb[:,1:]
            out_prev = mb[:,0:-1]
            costs = costs + model.train(inp, out_prev, out) ; N = N + 1
            print epoch, j, costs / N
            if j % 50 == 0:
                pred = model.predict(inp, out_prev)
                for i in range(len(pred)):
                    orig = tokenizer.inverse_transform([inp[i]])[0]
                    res =  tokenizer.inverse_transform([numpy.argmax(pred, axis=2)[i]])[0]
                    print orig
                    print res
        pickle.dump(model.network.params, open('params.{0}.pkl'.format(epoch),'w'))
    
if __name__ == '__main__':
    main()
