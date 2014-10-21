# Based upon code from Alessandro Mariani on Kaggle:
# http://www.kaggle.com/c/accelerometer-biometric-competition/forums/t/5401/how-do-you-cross-validate

import pickle
import numpy as np

from device_metadata import device_ids

def trim_mean(x, trim=10):
    lower_bound = np.percentile(x, trim)
    upper_bound = np.percentile(x, (100-trim))
    return np.mean(x[(x>=lower_bound) & (x<=upper_bound)])

data = pickle.load(open('train.pkl', 'rb'))

dev_steps = {}
for dev_id in device_ids:
    T = data[dev_id][:,0]
    steps = T[1:] - T[:-1]
    dev_steps[dev_id] = trim_mean(steps)

print '''
## SIMILAR DEVICE as per cluster_similar_device.py

devices_clusters = {
'''

for dev_id in device_ids:
    s = dev_steps[dev_id]
    window = 0.02
    while True:
        minval = s * (1 - window)
        maxval = s * (1 + window)
        similars = [ x for x in device_ids if minval <= dev_steps[x] <= maxval ]
        if len(similars) >= 4:
            break
        window *= 1.5
    print '  '+str(dev_id)+': '+str(similars)+','

print '}'
