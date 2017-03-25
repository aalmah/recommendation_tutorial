import numpy as np
import theano
import theano.tensor as T

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)
    
def normal(shape, mean=0., stdv=0.01):
    return np.random.normal(mean, stdv, size=shape)

def l2_norm(X):
    return T.sum(T.sqr(X))
