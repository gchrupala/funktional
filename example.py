# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
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
import copy
from layer import *

class EncoderDecoder(Layer):
    """A simple encoder-decoder net with shared input and output vocabulary."""
    def __init__(self, size_vocab, size, depth):
        self.size_vocab  = size_vocab
        self.size     = size
        self.depth    = depth
        self.Embed    = Embedding(self.size_vocab, self.size)
        self.Unembed  = self.Embed.unembed
        encoder = lambda size_in, size: StackedGRUH0(size_in, size, self.depth)
        decoder = lambda size_in, size: StackedGRU(size_in, size, self.depth)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Out      = Dense(size_in=self.size, size_out=self.size)
        self.params   = self.Embed.params + self.Encdec.params + self.Out.params
        
    def __call__(self, inp, out_prev):
        return softmax3d(self.Unembed(self.Out(self.Encdec(self.Embed(inp), self.Embed(out_prev)))))

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

def pad(xss, padding):
    max_len = max((len(xs) for xs in xss))
    def pad_one(xs):
        return xs + [ padding for _ in range(0,(max_len-len(xs))) ]
    return [ pad_one(xs) for xs in xss ]

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF G
    args = [iter(iterable)] * n
    return itertools.izip(*args)

def batch(item, BEG, END):
    """Prepare minibatch."""
    mb = numpy.array(pad([[BEG]+s+[END] for s in item], END), dtype='int32')
    inp = mb[:,1:]
    out = mb[:,1:]
    out_prev = mb[:,0:-1]
    return (inp, out_prev, out)

def valid_loss(model, valid, BEG, END):
    costs = 0.0; N = 0
    for _j, item in enumerate(grouper(valid, 256)):
        j = _j + 1
        inp, out_prev, out = batch(item, BEG, END)
        costs = costs + model.loss(inp, out_prev, out) ; N = N + 1
    return costs / N

def shuffled(x):
    y = copy.copy(x)
    random.shuffle(y)
    return y

def main():
    parser = argparse.ArgumentParser(description="Stacked recurrent autoencoder for sentences.")
    parser.add_argument('--size',   type=int, default=512,       help="Size of embeddings and hidden layers")
    parser.add_argument('--depth',  type=int, default=2,         help="Number of hidden layers")
    parser.add_argument('--epochs', type=int, default=1,         help="Number of training epochs")
    parser.add_argument('--seed',   type=int, default=None,      help="Random seed")
    parser.add_argument('--log',    type=str, default='log.txt', help="Path to log file")
    parser.add_argument('train',    type=str,                    help="Path to training data")
    parser.add_argument('valid',    type=str,                    help="Path to validation data")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    mapper = util.IdMapper(min_df=10)
    sents = shuffled(list(mapper.fit_transform([ line.split() for line in open(args.train) ])))
    sents_valid = list(mapper.transform([line.split() for line in open(args.valid) ]))
    mb_size = 128
    model = Model(size_vocab=mapper.size(), size=args.size, depth=args.depth)
    with open(args.log,'w') as log:
        for epoch in range(1,args.epochs + 1):
            costs = 0 ; N = 0
            for _j, item in enumerate(grouper(sents, 128)):
                j = _j + 1
                inp, out_prev, out = batch(item, mapper.BEG_ID, mapper.END_ID)
                costs = costs + model.train(inp, out_prev, out) ; N = N + 1
                print epoch, j, "train", costs / N
                if j % 500 == 0:
                    cost_valid = valid_loss(model, sents_valid, mapper.BEG_ID, mapper.END_ID)
                    print epoch, j, "valid", cost_valid
                if j % 100 == 0:
                    pred = model.predict(inp, out_prev)
                    for i in range(len(pred)):
                        orig = [ w for w in list(mapper.inverse_transform([inp[i]]))[0] 
                                 if w != mapper.END ]
                        res =  [ w for w in list(mapper.inverse_transform([numpy.argmax(pred, axis=2)[i]]))[0] 
                                 if w != mapper.END ]
                        log.write("{}".format(' '.join(orig)))
                        log.write("\n")
                        log.write("{}".format(' '.join(res)))
                        log.write("\n")
                    log.flush()
            pickle.dump(model, gzip.open('model.{0}.pkl.gz'.format(epoch),'w'))
    
if __name__ == '__main__':
    main()
