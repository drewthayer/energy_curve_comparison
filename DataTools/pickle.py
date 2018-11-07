import pickle
import os

def save_to_pickle(data, dir, fname):
    with open(os.path.join(dir, fname), 'wb') as f:
        pickle.dump(data, f)
    print('\n{} written to pkl\n'.format(fname))

def load_from_pickle(dir, fname):
    with open(os.path.join(dir,fname), 'rb') as f:
        data = pickle.load(f)
    return data
