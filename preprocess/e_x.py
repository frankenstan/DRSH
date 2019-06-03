import numpy as np
import h5py as h5
import json
a = h5.File('NUSatt_500.h5', 'r+')
b = json.load(open('/home/frx/mil/imgs_top10_label_train_test.json', 'r'))
c = json.load(open('/home/jzhou/ajb/glove300_dict.json', 'r'))
print len(c.keys())
d = h5.File('NUSgtGlove.h5', 'w')

lis = a.keys()
for k in range(len(a.keys())):
    idx = lis[k]
    attribute = a[idx][:]
    base = np.ones((1, 300))
    arg = attribute.argsort().argsort()
    for q in range(500):
        if arg[q] in np.array([499, 498, 497, 496, 495, 494, 493, 492, 491, 490]):
            word = b['all_label'][q]
            gloveVec = c[word]
            gloveVec = np.array(gloveVec)
            gloveVec = np.array([gloveVec])
            base = np.concatenate((base, gloveVec))

    base = np.delete(base, 0, 0)
    d.create_dataset(idx, data=base)
