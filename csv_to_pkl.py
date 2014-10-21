import pandas as pd
import pickle

def csv_to_pkl(csv_fn, group_key, pkl_fn):
    print 'Reading', csv_fn
    data = pd.read_csv(csv_fn)
    print 'Grouping'
    g = data.groupby(group_key)
    print 'Collecting'
    by_key = {}
    for key in g.groups.keys():
        by_key[key] = g.get_group(key).as_matrix()[:,:4]
    print 'Writing', pkl_fn
    pickle.dump(by_key, open(pkl_fn, 'wb'), -1)

csv_to_pkl('test.csv', 'SequenceId', 'test.pkl')
csv_to_pkl('train.csv', 'Device', 'train.pkl')
