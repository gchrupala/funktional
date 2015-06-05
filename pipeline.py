from util import autoassign

class NullModel(object):

    def fit(self):
        return self

    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

null_model = NullModel()

class Pipeline(object):
    '''Pipeline groups together the models needed to map sentences to vectors and back.'''

    def __init__(self, network, mapper=null_model, scaler=null_model):
        autoassign(locals())
    
    def project(self, sentences):
        '''Project sentences to vectors.'''
        inputs    = self.mapper.transform(sentences)
        out_v, _ = self.network(inputs)
        return self.scaler.inverse_transform(out_v)
