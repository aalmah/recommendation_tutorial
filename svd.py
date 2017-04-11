import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from utils import sharedX, normal, l2_norm, dump_params
from lasagne.updates import rmsprop, apply_momentum, sgd
from data_iterator import MultiFixDimIterator
import cPickle as pkl
import argparse
import os


def init_params(options):
    params = OrderedDict()
    params['W_users'] = sharedX(normal((options['n_users'],options['n_factors'])),
                                name='W_users')
    params['W_items'] = sharedX(normal((options['n_items'],options['n_factors'])),
                                name='W_items')
    params['b_users'] = sharedX(np.zeros((options['n_users'], 1)), name='b_users')
    params['b_items'] = sharedX(np.zeros((options['n_items'], 1)), name='b_items')
    params['b'] = sharedX(np.zeros(1), name='b')
    return params


def get_predictons(options, params, users_id, items_id):
    users_emb = params['W_users'][users_id]
    items_emb = params['W_items'][items_id]
    users_b   = params['b_users'][users_id]
    items_b   = params['b_items'][items_id]
    b         = params['b']
    rating = T.sum(users_emb*items_emb, axis=1, keepdims=True) + users_b + items_b + b
    return T.flatten(rating)


def build_model(options, params):
    # inputs to the model
    users_id = T.ivector('users_id')
    items_id = T.ivector('items_id')
    y        = T.fvector('y')

    # predictons
    y_hat = get_predictons(options, params, users_id, items_id)
    # cost
    mse = T.mean(T.sqr(y - y_hat))
    cost = mse
    if 'l2_coeff' in options and options['l2_coeff'] > 0.:
        cost += options['l2_coeff'] * sum([l2_norm(p) for p in params.values()])

    return users_id, items_id, y, y_hat, mse, cost


def prepare_full_data(raw_data):
    """Returns numpy ndarrays of user ids, item ids and ratings from raw_data"""
    users_id = np.asarray(raw_data[0], dtype='int32')
    items_id = np.asarray(raw_data[1], dtype='int32')
    ratings  = np.asarray(raw_data[3], dtype=theano.config.floatX)
    return [users_id, items_id, ratings]


def load_data(data_path):
    with open(data_path) as f:
        data_dict = pkl.load(f)
    n_users = data_dict['n_users']
    n_items = data_dict['n_products']
    train_data = prepare_full_data(data_dict["train"])
    valid_data = prepare_full_data(data_dict["valid"])
    test_data  = prepare_full_data(data_dict["test"])
    del data_dict
    return n_users, n_items, train_data, valid_data, test_data


def train(options, train_data, valid_data, test_data):
    np.random.seed(12345)

    if not os.path.exists(options['saveto']):
        os.makedirs(options['saveto'])

    print 'Building the model...'
    params = init_params(options)
    users_id, items_id, y, y_hat, mse, cost = build_model(options, params)

    print 'Computing gradients...'
    lrt = sharedX(options['lr'])
    grads = T.grad(cost, params.values())
    updates = sgd(grads, params.values(), lrt)
    updates = apply_momentum(updates, params.values(), momentum=options['momentum'])

    print 'Compiling theano functions...'
    eval_fn = theano.function([users_id, items_id, y], mse)
    train_fn = theano.function([users_id, items_id, y], [cost, mse],
                               updates=updates)

    print "Training..."
    train_iter = MultiFixDimIterator(*train_data, batch_size=options['batch_size'],
                                     shuffle=True)
    valid_iter = MultiFixDimIterator(*valid_data, batch_size=100)
    test_iter  = MultiFixDimIterator(*test_data,  batch_size=100)
    best_valid = None

    n_batches = np.ceil(train_data[0].shape[0]*1./options['batch_size']).astype('int')
    disp_str = ['COST', 'Train MSE', 'Valid MSE', 'Test MSE']

    for eidx in range(options['n_epochs']):
        accum_mse, accum_cost = 0., 0.
        for batch in train_iter:
            b_cost, b_mse = train_fn(*batch)
            accum_cost += b_cost
            accum_mse  += b_mse

        disp_val = [val/n_batches for val in [accum_cost, accum_mse]]
        disp_val += [np.mean([eval_fn(*batch) for batch in valid_iter]),
                     np.mean([eval_fn(*batch) for batch in test_iter])]
        if best_valid is None or best_valid > disp_val[0]:
            best_valid = disp_val[0]
            dump_params(options['saveto'], eidx, "best_params", params)

        res_str = ('[%d] ' % eidx) + ", ".join("%s: %.4f" %(s,v) for s,v in
                                               zip(disp_str, disp_val))
        print res_str

    print "Done"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/arts.pkl', help='path to processed data (pickle file)')
    parser.add_argument('--n_factors', type=int, default=5, help='number of hidden factors')
    parser.add_argument('--l2_coeff', type=float, default=0.0, help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=128, help='size of training batch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--saveto', type=str, default="svd_model", help='path to save best model')
    options = vars(parser.parse_args())

    print "Loading data..."
    n_users, n_items, train_data, valid_data, test_data = load_data(options['data_path'])
    options['n_users'] = n_users
    options['n_items'] = n_items
    train(options, train_data, valid_data, test_data)
