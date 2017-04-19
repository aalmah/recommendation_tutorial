"""Bag-of-Words Latent Factor Model or (BoWLF)"""

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from utils import sharedX, normal, l2_norm, sgd, dump_params
from data_iterator import MultiFixDimIterator
import cPickle as pkl
import argparse
import os


def init_params(options):
    params = OrderedDict()
    # LF model params
    params['W_users'] = sharedX(normal((options['n_users'],options['n_factors'])),
                                name='W_users')
    params['W_items'] = sharedX(normal((options['n_items'],options['n_factors'])),
                                name='W_items')
    params['b_users'] = sharedX(np.zeros((options['n_users'],)), name='b_users')
    params['b_items'] = sharedX(np.zeros((options['n_items'],)), name='b_items')
    params['b'] = sharedX(0., name='b')

    # distributed BOW params
    params['W_bow'] = sharedX(normal((options['n_factors'],options['vocab_size'])),
                              name='W_bow')
    params['b_bow'] = sharedX(np.zeros((options['vocab_size'],)), name='b_bow')
    return params


def get_predictons(options, params, users_id, items_id):
    # first part is the same as SVD model
    users_emb = params['W_users'][users_id]
    items_emb = params['W_items'][items_id]
    users_b   = params['b_users'][users_id]
    items_b   = params['b_items'][items_id]
    b         = params['b']
    rating = T.sum(users_emb*items_emb, axis=1) + users_b + items_b + b

    # predict review text BoW from products embedding
    bow = T.nnet.softmax(T.dot(items_emb, params['W_bow']) + params['b_bow'])

    return T.flatten(rating), bow


def build_model(options, params):
    # inputs to the model
    users_id   = T.ivector('users_id')
    items_id   = T.ivector('items_id')
    bow        = T.fmatrix('bow')
    y          = T.fvector('y')

    alpha = options['alpha']

    # predictons
    y_pred, bow_pred = get_predictons(options, params, users_id, items_id)
    # LF model cost
    mse = T.mean(T.sqr(y - y_pred))

    # BOW negative-log-likelihood cost
    nll = T.mean(T.nnet.categorical_crossentropy(bow_pred, bow))

    cost = alpha * mse + (1-alpha) * nll

    if 'l2_coeff' in options and options['l2_coeff'] > 0.:
        cost += options['l2_coeff'] * sum([l2_norm(p) for p in params.values()])

    return users_id, items_id, bow, y, y_pred, bow_pred, mse, nll, cost


def prepare_full_data(raw_data, vocab_size, return_reviews=True):
    """Returns numpy ndarrays of user ids, item ids, review lists, and ratings from raw_data"""
    users_id = np.asarray(raw_data[0], dtype='int32')
    items_id = np.asarray(raw_data[1], dtype='int32')
    ratings  = np.asarray(raw_data[3], dtype=theano.config.floatX)
    reviews_list = np.asarray(raw_data[2])  # reviews as ndarrays of lists
    if return_reviews:
        return [users_id, items_id, reviews_list, ratings]
    else:
        return [users_id, items_id, ratings]


def prepare_batch_data(options, batch):
    uid, iid, revlist, ratings = batch
    rev_bow = np.zeros((len(revlist), options['vocab_size']),
                       dtype=theano.config.floatX)
    for i,rlist in enumerate(revlist):
        for word in rlist:
            rev_bow[i][word] = rev_bow[i][word] + 1.
    # set 0 and 1 token counts to 1, so they don't affect result
    rev_bow[:,[0,1]] = 1.
    rev_bow = rev_bow / rev_bow.sum(axis=1, keepdims=True)
    return (uid, iid, rev_bow, ratings)


def load_data(data_path):
    with open(data_path) as f:
        data_dict = pkl.load(f)
    n_users = data_dict['n_users']
    n_items = data_dict['n_products']
    train_data = prepare_full_data(data_dict["train"], options['vocab_size'])
    valid_data = prepare_full_data(data_dict["valid"], options['vocab_size'], return_reviews=False)
    test_data  = prepare_full_data(data_dict["test"], options['vocab_size'], return_reviews=False)
    del data_dict
    return n_users, n_items, train_data, valid_data, test_data


def train(options, train_data, valid_data, test_data):
    np.random.seed(12345)

    if not os.path.exists(options['saveto']):
        os.makedirs(options['saveto'])

    print 'Building the model...'
    params = init_params(options)
    users_id, items_id, bow, y, y_pred, bow_pred, mse, nll, cost = build_model(options, params)

    print 'Computing gradients...'
    lrt = sharedX(options['lr'])
    grads = T.grad(cost, params.values())
    updates = sgd(params.values(), grads, lrt)

    print 'Compiling theano functions...'
    eval_fn = theano.function([users_id, items_id, y], mse)
    train_fn = theano.function([users_id, items_id, bow, y], [cost, mse, nll],
                               updates=updates)

    print "Training..."
    train_iter = MultiFixDimIterator(*train_data, batch_size=options['batch_size'],
                                     shuffle=True)
    valid_iter = MultiFixDimIterator(*valid_data, batch_size=100)
    test_iter  = MultiFixDimIterator(*test_data,  batch_size=100)
    best_valid = float('inf')
    best_test  = float('inf')

    n_batches = np.ceil(train_data[0].shape[0]*1./options['batch_size']).astype('int')
    disp_str = ['Train COST', 'Train MSE', 'Train NLL']

    for eidx in range(options['n_epochs']):
        accum_cost, accum_mse, accum_nll = 0., 0., 0.
        for batch in train_iter:
            batch = prepare_batch_data(options, batch)
            b_cost, b_mse, b_nll = train_fn(*batch)
            accum_cost += b_cost
            accum_mse  += b_mse
            accum_nll  += b_nll

        disp_val = [val/n_batches for val in [accum_cost, accum_mse, accum_nll]]
        res_str = ('[%d] ' % eidx) + ", ".join("%s: %.4f" %(s,v) for s,v in
                                               zip(disp_str, disp_val))
        print res_str

        if (eidx+1) % options['valid_freq'] == 0:
            disp_val = [np.mean([eval_fn(*vbatch) for vbatch in valid_iter]),
                        np.mean([eval_fn(*tbatch) for tbatch in test_iter])]
            res_str = ", ".join("%s: %.4f" %(s,v) for s,v in
                                zip(['Valid MSE', 'Test MSE'], disp_val))
            print res_str

            if best_valid > disp_val[0]:
                best_valid, best_test = disp_val
                dump_params(options['saveto'], eidx, "best_params", params)

    print "Done training..."
    print "Best Valid MSE: %.4f and Test MSE: %.4f" % best_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/arts.pkl', help='path to processed data (pickle file)')
    parser.add_argument('--n_factors', type=int, default=5, help='number of hidden factors')
    parser.add_argument('--vocab_size', type=int, default=5000, help='number of hidden factors')
    parser.add_argument('--alpha', type=float, default=0.08, help='weight decay coefficient')
    parser.add_argument('--l2_coeff', type=float, default=0.0, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=128, help='size of training batch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--valid_freq', type=int, default=2, help='validation frequency (in epochs)')
    parser.add_argument('--saveto', type=str, default="bowlf_model", help='path to save best model')
    options = vars(parser.parse_args())

    print "Loading data..."
    n_users, n_items, train_data, valid_data, test_data = load_data(options['data_path'])
    options['n_users'] = n_users
    options['n_items'] = n_items
    train(options, train_data, valid_data, test_data)
