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
from imaginet import Imaginet, MultitaskLM, MultitaskED, predictor_v
import evaluate
import json

def batch_inp(sents, BEG, END):
    mb = padder(sents, BEG, END)
    return mb[:,1:]

def padder(sents, BEG, END):
    return numpy.array(pad([[BEG]+sent+[END] for sent in sents], END), dtype='int32')

def batch(item, BEG, END):
    """Prepare minibatch."""
    mb_inp = padder([s for s,_,_ in item])
    mb_out_t = padder([r for _,r,_ in item])
    inp = mb_inp[:,1:]
    out_t = mb_out_t[:,1:]
    out_prev_t = mb_out_t[:,0:-1]
    out_v = numpy.array([ t for _,_,t in item ], dtype='float32')
    return (inp, out_v, out_prev_t, out_t)
    
def valid_loss(model, sents_val_in, sents_val_out, images_val, BEG_ID, END_ID,
               batch_size=128):
    """Apply model to validation data and return loss info."""
    triples = zip(sents_val_in, sents_val_out, images_val)
    c = Counter()
    for item in grouper(triples, batch_size):
        inp, out_v, out_prev_t, out_t = batch_imaginet(item, BEG_ID, END_ID)
        cost, cost_t, cost_v = model.loss(inp, out_v, out_prev_t, out_t)
        c += Counter({'cost_t': cost_t, 'cost_v': cost_v, 'cost': cost, 'N': 1})
    return c
    
class NoScaler():
    def __init__(self):
        pass
    def fit_transform(self, x):
        return x
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

def stats(c):
    return " ".join(map(str, [c['cost_t']/c['N'], c['cost_v']/c['N'], c['cost']/c['N']]))

def cmd_train( dataset='coco',
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
               validate_period=64*100,
               logfile='log.txt'):
    sys.setrecursionlimit(50000) # needed for pickling models
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
    pickle.dump(scaler, gzip.open(os.path.join(model_path, 'scaler.pkl.gz'), 'w'),
                protocol=pickle.HIGHEST_PROTOCOL)
    sents_in      = list(mapper.fit_transform(sents))
    sents_out     = list(mapper.transform(sents))

    
    sents_val_in  = list(mapper.transform(sents_val))
    sents_val_out = list(mapper.transform(sents_val))

    pickle.dump(mapper, gzip.open(os.path.join(model_path, 'mapper.pkl.gz'),'w'),
                protocol=pickle.HIGHEST_PROTOCOL)
    model = Imaginet(size_vocab=mapper.size(),
                     size_embed=embedding_size,
                     size=hidden_size,
                     size_out=4096,
                     depth=depth,
                     network=architecture,
                     alpha=alpha)
    triples = zip(sents_in, sents_out, images)
    with open(logfile, 'w') as log:
        for epoch in range(1, epochs + 1):
            costs = Counter()
            N = 0
            for _j, item in enumerate(grouper(triples, batch_size)):
                j = _j + 1
                inp, out_v, out_prev_t, out_t = batch_imaginet(item, mapper.BEG_ID, mapper.END_ID)
                cost, cost_t, cost_v = model.train(inp, out_v, out_prev_t, out_t)
                costs += Counter({'cost_t':cost_t, 'cost_v': cost_v, 'cost': cost, 'N': 1})
                print epoch, j, j*batch_size, "train", stats(costs)
                if j*batch_size % validate_period == 0:
                    costs_valid = valid_loss(model, sents_val_in, sents_val_out, images_val,
                                             mapper.BEG_ID, mapper.END_ID)
                    print epoch, j, j, "valid", stats(costs_valid)
                sys.stdout.flush()
            pickle.dump(model, gzip.open(os.path.join(model_path, 'model.{0}.pkl.gz'.format(epoch)),'w'))
        pickle.dump(model, gzip.open(os.path.join(model_path, 'model.pkl.gz'), 'w'))
        
def cmd_predict_v(dataset='coco',
                  model_path='.',
                  batch_size=128,
                  output='predict_v.npy'):
    def load(f):
        return pickle.load(gzip.open(os.path.join(model_path, f)))
    mapper, scaler, model = map(load, ['mapper.pkl.gz','scaler.pkl.gz','model.pkl.gz'])
    predict_v = predictor_v(model)
    prov   = dp.getDataProvider(dataset)
    sents  = list(prov.iterSentences(split='val'))
    inputs = list(mapper.transform([sent['tokens'] for sent in sents ]))
    preds  = numpy.vstack([ predict_v(batch_inp(batch, mapper.BEG_ID, mapper.END_ID))
                            for batch in grouper(inputs, batch_size) ])
    numpy.save(os.path.join(model_path, output), preds)
    
def cmd_eval(dataset='coco',
             scaler_path='scaler.pkl.gz',
             input='predict_v.npy',
             output='eval.json'):
    scaler = pickle.load(gzip.open(scaler_path))
    preds  = numpy.load(input)
    prov   = dp.getDataProvider(dataset)
    sents  = list(prov.iterSentences(split='val'))
    images = list(prov.iterImages(split='val'))
    img_fs = list(scaler.transform([ image['feat'] for image in images ]))
    correct = numpy.array([ [ sents[i]['imgid']==images[j]['imgid']
                              for j in range(len(images)) ]
                            for i in range(len(sents)) ])
    r = evaluate.ranking(img_fs, preds, correct, exclude_self=False)
    json.dump(r, open(output, 'w'))
    
