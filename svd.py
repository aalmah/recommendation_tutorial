import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from utils import sharedX, normal, l2_norm
from lasagne.updates import rmsprop, apply_momentum
from data_iterator import MultiFixDimIterator
import cPickle as pkl
from tqdm import tqdm
import argparse

def init_params(options):
    params = OrderedDict()
    params['W_users'] = sharedX(normal((options['n_users'],options['n_factors'])),
                                name='W_users')
    params['W_items'] = sharedX(normal((options['n_items'],options['n_factors'])),
                                name='W_items')
    params['b_users'] = sharedX(np.zeros((options['n_users'], 1)), name='b_users')
    params['b_items'] = sharedX(np.zeros((options['n_items'], 1)), name='b_items')
    params['b'] = sharedX(np.zeros((options['n_items'], 1)), 'b')
    return params


def get_embedding(input, W, emb_size):
    out_shape = [input.shape[i] for i in range(input.ndim)] + [emb_size]
    return W[input.flatten()].reshape(out_shape)


def get_predictons(options, params, users_id, items_id):
    users_emb = get_embedding(users_id, params['W_users'], options['n_factors'])
    items_emb = get_embedding(items_id, params['W_items'], options['n_factors'])
    users_b   = get_embedding(users_id, params['b_users'], 1)
    items_b   = get_embedding(items_id, params['b_items'], 1)
    b         = params['b']
    return T.sum(users_emb*items_emb, axis=1, keepdims=True) + users_b + items_b + b


def build_model(options, params):
    # inputs to the model
    users_id = T.imatrix('users_id')
    items_id = T.imatrix('items_id')
    y        = T.matrix('y')

    # predictons
    y_hat = get_predictons(options, params, users_id, items_id)

    # cost
    mse = T.mean(T.sqr(y_hat - y))
    cost = mse
    if 'l2_coeff' in options and options['l2_coeff'] > 0.:
        cost += options['l2_coeff'] * sum([l2_norm(p) for p in params.values()])

    return users_id, items_id, y, y_hat, mse, cost


def prepare_data(raw_data):
    """Returns numpy ndarrays of user ids, item ids and ratings from raw_data"""
    users_id = np.asarray(raw_data[0], dtype='int64')
    items_id = np.asarray(raw_data[1], dtype='int64')
    ratings  = np.asarray(raw_data[3], dtype=theano.config.floatX)
    return [users_id, items_id, ratings]


def load_data(data_path):
    with open(data_path) as f:
        data_dict = pkl.load(f)
    n_users = data_dict['n_users']
    n_items = data_dict['n_products']
    train_data = prepare_data(data_dict["train"])
    valid_data = prepare_data(data_dict["valid"])
    test_data = prepare_data(data_dict["test"])
    del data_dict
    return n_users, n_items, train_data, valid_data, test_data


def train(options, train_data, valid_data, test_data):
    np.random.seed(12345)

    print 'Building the model...'
    params = init_params(options)
    users_id, items_id, y, y_hat, mse, cost = build_model(options, params)

    print 'Computing gradients...'
    lrt = sharedX(options['lr'])
    grads = T.grad(cost, params.values())
    updates = rmsprop(grads, params.values(), lrt)
    updates = apply_momentum(updates, params.values(), momentum=options['momentum'])

    print 'Compiling theano functions...'
    eval_fn = theano.function([users_id, items_id, y], mse)
    train_fn = theano.function([users_id, items_id, y], [cost, mse],
                               updates=updates)

    print "Training..."
    train_iter = MultiFixDimIterator(train_data, options['batch_size'],
                                     shuffle=True)
    valid_iter = MultiFixDimIterator(valid_data, 100)
    test_iter  = MultiFixDimIterator(test_data,  100)

    n_batches = np.ceil(train_data[0].shape[0]*1./options['batch_size']).astype('int')
    accum_mse, accum_cost = 0., 0.
    disp_str = ['COST', 'Train MSE', 'Valid MSE', 'Test MSE']

    for eidx in range(options['n_epochs']):
        for uidx, batch in tqdm(enumerate(train_iter), total=n_batches, ncols=160):
            b_cost, b_mse = train_fn(*batch)
            accum_cost += b_cost
            accum_mse  += b_mse

        disp_val = [val/n_batches for val in [accum_cost, accum_mse]]
        disp_val += [np.mean([eval_fn(*batch) for batch in valid_iter]),
                     np.mean([eval_fn(*batch) for batch in test_iter])]
        res_str = ('[%d] ' % eidx) + ", ".join("%s: %.2f" %(s,v) for s,v in
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
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    options = vars(parser.parse_args())

    print "Loading data..."
    n_users, n_items, train_data, valid_data, test_data = load_data(options['data_path'])
    options['n_users'] = n_users
    options['n_items'] = n_items
    train(options, train_data, valid_data, test_data)
