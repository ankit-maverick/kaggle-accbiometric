#!/usr/bin/python

# Get sensor characteristics for each device.

from __future__ import division

import numpy as np
import pickle
import gzip
import random

# Read data files (if not already loaded)
if not 'train' in locals():
    print 'Reading train.pkl...'
    train = pickle.load(open('train.pkl', 'rb'))
    print '... done'

def approx_gcd(a,b):
    assert a > 0 and b > 0
    while True:
        if a < b:
            (a,b) = (b,a)
        a = a % b
        if a < 1e-3:
            return b

def get_resolution(x):
    #xm = np.median(x)
    #xg = x[(x > xm-2) & (x < xm+2)]
    xg = x.copy()
    xg -= np.min(xg)
    xg = xg[xg > 0]
    v = np.array([ approx_gcd(random.choice(xg), random.choice(xg)) for i in range(1000) ])
    v = v[np.isfinite(v)]
    m = np.median(v)
    #return np.mean(v[(v > m*0.8) & (v < m*1.2)])
    return m

def get_characteristics(seq):
    (T, X, Y, Z) = seq.T
    dt = np.array(sorted(T[1:] - T[:-1]))
    dt_dupes = np.sum(dt == 0) / len(dt)
    dt = dt[(dt > 0) & (dt < 1000)]
    return {
        'res': get_resolution(X),
        'dt_dupes': dt_dupes,
        'dt_10pct': dt[len(dt)*0.1],
        'dt_90pct': dt[len(dt)*0.9],
        'dt_mean': np.mean(dt[len(dt)*0.1:len(dt)*0.9]),
        'dt_median': np.median(dt),
        'nsamps': seq.shape[0],
    }

stats = {}
for dev_id in train.keys():
    if dev_id % 10 == 0:
        print dev_id
    stats[dev_id] = get_characteristics(train[dev_id])
pickle.dump(stats, gzip.open('device_characteristics.pkl.gz', 'wb'), -1)
