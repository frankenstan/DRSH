import numpy as np
import h5py as h5

a = h5.File('NUSgtGlove.h5', 'r')
b = h5.File('NUSGlove.h5', 'w')

ids = np.load('id.npy')


ak = a.keys()
for k in range(269648):
    keyy = ids[k]
    if keyy not in ids[:k]:
        b.create_dataset(keyy, data=a[ak[k]][:])
