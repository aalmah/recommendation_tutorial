import gzip
import numpy as np
import cPickle as pkl
from subprocess import Popen, PIPE
import os
import argparse
import time
import urllib

VERBOSE = True

def parse(filename):
    f = gzip.open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(':')
        if colonPos == -1:
            yield entry
            entry = {}
            continue
        eName = l[:colonPos]
        rest = l[colonPos + 2:]
        entry[eName] = rest
    yield entry


def generate_data_list(data_path):
    """Generate list with each row has a transaction information"""
    product_dict = {}
    product_titles = {}
    user_dict = {}
    data = []
    for e in parse(data_path):
        if e == {}:
            continue
        if e["product/productId"] not in product_dict:
            product_id = len(product_dict)
            product_dict[e["product/productId"]] = product_id
            product_titles[product_id] = e["product/title"]
        if e["review/userId"] not in user_dict:
            user_id = len(user_dict)
            user_dict[e["review/userId"]] = user_id
        row = []
        row.append(user_id)
        row.append(product_id)
        row.append(e["review/text"])
        row.append(float(e["review/score"]))
        data.append(row)

    return data, user_dict, product_dict, product_titles


def split_data(data):
    """Make a split assuming data is shuffled"""
    train_len = int(len(data) * .8)
    valid_len = (len(data) - train_len) / 2
    train_data = data[:train_len]
    valid_data = data[train_len:train_len + valid_len]
    test_data = data[train_len + valid_len:]
    return train_data, valid_data, test_data


def tokenize(sentences):
    tokenizer_cmd = ['tokenizer.perl', '-l', 'en', '-q', '-']
    if VERBOSE:
        print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    if VERBOSE:
        print 'Done'

    return toks


def build_dict(train_data, max_n):
    users, products, sentences, labels = zip(*train_data)
    sentences = tokenize(sentences)
    if VERBOSE:
        print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1
    counts = wordcount.values()
    keys = wordcount.keys()
    sorted_idx = np.argsort(counts)[::-1]
    worddict = dict()
    for idx, ss in enumerate(sorted_idx[:max_n-2]):
        worddict[keys[ss]] = idx + 2  # leave 0 and 1 (UNK)
    if VERBOSE:
        print '# words: {}, # unique words: {}'.format(np.sum(counts), len(keys))
    return worddict


def sort_data(data):
    sorted_data = [t[1] for t in sorted([(len(s), s)for s in data])]
    return sorted_data


def preprocess_data(data, dictionary, sort=False):
    users, products, sentences, labels = zip(*data)
    sentences_tok = tokenize(sentences)
    assert len(sentences_tok) == len(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences_tok):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]
        if len(seqs[idx]) == 0:
            if VERBOSE:
                print 'empty review.. %d' % idx
            seqs[idx] = [1]

    if sort:
        seqs = sort_data(seqs)
    return (users, products, seqs, labels)


def main(category, data_path, vocab_size):

    t1 = time.time()
    np.random.seed(12345)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    fname = '{}.txt.gz'.format(category.title())
    fpath = os.path.join(data_path, fname)
    if not os.path.isfile(fpath):
        print "Data is not available in the path: '{}', downloading...".format(data_path)
        urllib.urlretrieve("https://snap.stanford.edu/data/amazon/{}".format(fname), fpath)
    data, user_dict, product_dict, product_titles = \
        generate_data_list(fpath)
    # shuffle data
    np.random.shuffle(data)

    n_users = len(user_dict)
    n_products = len(product_dict)

    if VERBOSE:
        print "# data: {}, # users: {}, # products: {}".format(len(data), n_users, n_products)

    train_data, valid_data, test_data = split_data(data)

    dictionary = build_dict(train_data, vocab_size)
    train_data = preprocess_data(train_data, dictionary)
    valid_data = preprocess_data(valid_data, dictionary)
    test_data = preprocess_data(test_data, dictionary)
    if VERBOSE:
        print "# train:", len(train_data[0])
        print "# valid:", len(valid_data[0])
        print "# test:", len(test_data[0])
        print "vocab_size:", len(dictionary)
    with open(os.path.join(data_path, '%s.pkl' % category), 'wb') as f:
        pkl.dump({"train": train_data,
                  "valid": valid_data,
                  "test": test_data,
                  "n_users": n_users,
                  "n_products": n_products,
                  "dictionary": dictionary}, f)
        # saving extra information, in case needed for debugging
        pkl.dump({"product_titles": product_titles,
                  "product_dict": product_dict,
                  "user_dict": user_dict}, f)
    if VERBOSE:
        t2 = time.time()
        print 'took %.4f minutes' % ((t2 - t1) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str,
                        default='arts', help='product category')
    parser.add_argument('-p', '--path', type=str,
                        default='data', help='path to data')
    parser.add_argument('-v', '--vocab_size', type=int, default=5000,
                        help='size of vocabulary of reviews')
    args = parser.parse_args()

    main(args.category, args.path, args.vocab_size)
