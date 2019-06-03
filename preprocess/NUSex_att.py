import numpy as np
import os
import h5py as h5
import sys
import scipy
from scipy import misc

caffe_root = '/home/frx/mil/milcode/caffe/'
sys.path.insert(0, caffe_root + 'python')
os.chdir('/home/frx/NUSImage/')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

model_def = '/home/frx/mil/vgg/mil_finetune.prototxt.deploy'
model_weights = '/home/frx/_iter_560000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', np.array([111.87271653357466, 119.73625525773066, 119.68077455437107]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          565, 565)  # image size is 224x224

a = h5.File('NUSatt.h5', 'w')
b = os.listdir('Flickr')
for k in range(len(b)):
    imgs_idx = os.listdir('Flickr/' + b[k])
    for i in range(len(imgs_idx)):
        if imgs_idx[i][-3:] == 'jpg':
            image = caffe.io.load_image('/home/frx/NUSImage/Flickr/' + b[k] + '/' + imgs_idx[i])
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            feature = output['mil'][0, :, 0, 0]
            feature = np.array(feature)
            a.create_dataset(imgs_idx[i], data=feature)
