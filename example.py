# A simple encoder-decoder example with funktional
from __future__ import division 
import theano
import numpy
import random
import itertools
import cPickle as pickle
import argparse
import gzip
import sys
import util
from layer import *

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
        # Like train, but no updates
        self.loss = theano.function([self.input, self.output_prev, self.output ], self.cost)

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

def batch(mapper, item):
    """Prepare minibatch."""
    mb = numpy.array(pad([[mapper.PAD_ID]+s+[mapper.END_ID] for s in item], mapper.PAD_ID), dtype='int32')
    inp = mb[:,1:]
    out = mb[:,1:]
    out_prev = mb[:,0:-1]
    return (inp, out_prev, out)

def main():
    parser = argparse.ArgumentParser(description="Stacked recurrent autoencoder for sentences.")
    parser.add_argument('--size', type=int, default=512, help="Size of embeddings and hidden layers")
    parser.add_argument('--depth', type=int, default=2, help="Number of hidden layers")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('train', type=str, help="Path to training data")
    parser.add_argument('valid', type=str, help="Path to validation data")
    args = parser.parse_args()
    mapper = util.IdMapper(min_df=10)
    sents = mapper.fit_transform([ line.split() for line in open(args.train) ])
    sents_valid = mapper.transform([line.split() for line in open(args.valid) ])
    batch_valid = batch(mapper, sents_valid)
    mb_size = 128
    model = Model(size_vocab=mapper.size(), size=args.size, depth=args.depth)
    for epoch in range(1,args.epochs + 1):
        costs = 0 ; N = 0
        for _j, item in enumerate(grouper(sents, 128)):
            j = _j + 1
            inp, out_prev, out = batch(mapper, item)
            costs = costs + model.train(inp, out_prev, out) ; N = N + 1
            print epoch, j, "train", costs / N
            if j % 50 == 0:
                loss_valid = model.loss(*batch_valid)
                print epoch, j, "valid", loss_valid
                pred = model.predict(inp, out_prev)
                for i in range(len(pred)):
                    orig = mapper.inverse_transform([inp[i]])[0]
                    res =  mapper.inverse_transform([numpy.argmax(pred, axis=2)[i]])[0]
                    sys.stderr.write(orig)
                    sys.stderr.write("\n")
                    sys.stderr.write(res)
                    sys.stderr.write("\n")
        pickle.dump(model, gzip.open('model.{0}.pkl.gz'.format(epoch),'w'))
    
if __name__ == '__main__':
    main()
