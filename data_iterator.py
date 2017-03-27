import numpy as np

class MultiFixDimIterator(object):
    """Iterate multiple fixed-dim ndarrays (e.g. inputs and labels) and return tuples of minibatches"""
    def __init__(self, *data, **kwargs):
        super(MultiFixDimIterator, self).__init__()

        assert all(d.shape[0] == data[0].shape[0] for d in data), 'data differs in number of instances!'
        self.data = data

        self.num_data = data[0].shape[0]

        batch_size = kwargs.get('batch_size', 100)
        shuffle = kwargs.get('shuffle', False)
        
        self.n_batches = self.num_data / batch_size
        if self.num_data % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()
    
    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_data)
        else:
            self.data_indices = np.arange(self.num_data)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1
        
        batch = tuple(data[chosen_indices] for data in self.data)
        return batch
    
if __name__ == "__main__":
    import cPickle as pkl
    data_dict = pkl.load(open('data/arts.pkl'))
    import pdb; pdb.set_trace()
    print "ok"
