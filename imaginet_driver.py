#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# Imaginet (http://arxiv.org/abs/1506.03694) models implemented with funktional
from __future__ import division 
import theano
import numpy
import random
import itertools
import cPickle as pickle
import argparse
import gzip
import sys
import os
import util
import copy
import time
from collections import Counter
import imagernn.data_provider as dp
from layer import *
from imaginet import Imaginet, MultitaskLM, MultitaskAE

def batch_imaginet(item, BEG, END):
    """Prepare minibatch."""
    mb_inp = numpy.array(pad([[BEG]+s+[END] for s,_,_ in item], END), dtype='int32')
    mb_out_t = numpy.array(pad([[BEG]+r+[END] for _,r,_ in item], END), dtype='int32')
    inp = mb_inp[:,1:]
    out_t = mb_out_t[:,1:]
    out_prev_t = mb_out_t[:,0:-1]
    out_v = numpy.array([ t for _,_,t in item ], dtype='float32')
    return (inp, out_v, out_prev_t, out_t)

class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

def train_cmd( dataset='coco',
               model_path='.',
               hidden_size=512,
               embedding_size=None,
               depth=1,
               scaler=None,
               seed=None,
               shuffle=False,
               architecture=MultitaskLM,
               alpha=0.1,
               epochs=1,
               batch_size=64,
               logfile='log.txt'):
               
    if seed is not None:
        random.seed(seed)
    prov = dp.getDataProvider(dataset)
    data = list(prov.iterImageSentencePair(split='train'))
    data_val = list(prov.iterImageSentencePair(split='val'))
    if shuffle:
        numpy.random.shuffle(data)
    mapper = util.IdMapper(min_df=10)
    sents, images_ = zip(*[ (pair['sentence']['tokens'], pair['image']['feat']) for pair in data ])
    embedding_size = embedding_size if embedding_size is not None else hidden_size
    scaler = StandardScaler() if scaler == 'standard' else NoScaler()

    sents_val, images_val_ = \
                    zip(*[ (pair['sentence']['tokens'], pair['image']['feat']) for pair in data_val ])

    images = scaler.fit_transform(images_)
    images_val = scaler.transform(images_val_)
    pickle.dump(scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    sents_in      = list(mapper.fit_transform(sents))
    sents_out     = list(mapper.transform(sents))

    
    sents_val_in  = list(mapper.transform(sents_val))
    sents_val_out = list(mapper.transform(sents_val))

    pickle.dump(mapper, gzip.open(os.path.join(model_path, 'mapper.pkl.gz'),'w'), protocol=pickle.HIGHEST_PROTOCOL)
    mb_size = 128
    model = Imaginet(size_vocab=mapper.size(),
                     size_embed=embedding_size,
                     size=hidden_size,
                     size_out=4096,
                     depth=depth,
                     network=architecture,
                     alpha=alpha)
    triples = zip(sents_in, sents_out, images_val)
    with open(logfile, 'w') as log:
        for epoch in range(1, epochs + 1):
            costs = Counter({'cost_t':0.0, 'cost_v': 0.0, 'cost': 0.0, 'N': 0})
            N = 0
            for _j, item in enumerate(grouper(triples, batch_size)):
                j = _j + 1
                inp, out_v, out_prev_t, out_t = batch_imaginet(item, mapper.BEG_ID, mapper.END_ID)
                cost, cost_t, cost_v = model.train(inp, out_v, out_prev_t, out_t)
                costs += Counter({'cost_t':cost_t, 'cost_v': cost_v, 'cost': cost, 'N': 1})
                print epoch, j, j*mb_size, "train",\
                          costs['cost_t']/costs['N'],\
                          costs['cost_v']/costs['N'],\
                          costs['cost']/costs['N']
                          
                # TODO run validation
            pickle.dump(model, gzip.open(os.path.join(model_path, 'model.{0}.pkl.gz'.format(epoch)),'w'))
        dump(model, gzip.open(os.path.join(model_path, 'model.pkl.gz'), 'w'))
        
def encode(model, mapper, sents):
    """Return projections of `sents` to the final hidden state of the encoder of `model`."""
    return numpy.vstack([  model.project(batch(item, mapper.BEG_ID, mapper.END_ID)[0]) 
                           for item in grouper(mapper.transform(sents), 128) ])
def encode_cmd(args):
    model = pickle.load(gzip.open(os.path.join(args.model_path, 'model.pkl.gz')))
    mapper = pickle.load(gzip.open(os.path.join(args.model_path, 'mapper.pkl.gz')))
    sents = [line.split() for line in open(args.input_file) ]
    pickle.dump(encode(model, mapper, sents), gzip.open(args.output_file, 'w'))
