import numpy as np
import pandas as pd

from device_metadata import device_ids

data = pd.HDFStore('train_by_dev.h5', 'r')

out = []

for dev_id in device_ids:
    dev_data = data['/'+str(dev_id)]
    nsamps = len(dev_data)
    out.append(nsamps)

data.close()

print 'train_lengths =', out
