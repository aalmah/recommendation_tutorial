import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def normal(shape, mean=0., stdv=0.01):
    return np.random.normal(mean, stdv, size=shape)


def l2_norm(X):
    return T.sum(T.sqr(X))


def sgd(params, grads, learning_rate):
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def dump_params(saveto, eidx, name, params):
    print "saving %s at epoch %d..." % (name, eidx)
    np.savez("%s/%s" % (saveto, name), epoch_idx=eidx, **params)
