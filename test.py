import theano
from theano import tensor as T
from theano import config
import numpy as np
import backend as K
import regularizers
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
import initializations_2
import initializations
import activations
from theano import config
import optimizer
import time
import cPickle as pkl
import h5py as h5
import json
from scipy.spatial.distance import hamming as ham
import sys
import random
from random import shuffle
sys.setrecursionlimit(1000000)
from collections import OrderedDict
from scipy import io as sio
from sklearn.decomposition import PCA
def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

f_init_ = initializations.get('glorot_uniform')
f_init = initializations_2.get('glorot_uniform')
f_init2 = initializations_2.get('uniform')
f_inner_init = initializations.get('orthogonal')
f_forget_bias_init = initializations.get('one')

def init_params(options):
    dim_frame = options['dim_frame']
    att_frame = options['att_frame']
    steps = options['steps']
    hidden_dim = options['hidden_dim']
    # batch_size = options['batch_size']

    params = OrderedDict()

    params['W_in'] = f_init_((att_frame, 4 * hidden_dim))

    params['W_fea'] = f_init_((dim_frame, 2 * hidden_dim))
    params['b_fea'] = K.zeros((1, 2 * hidden_dim))

    params['W_hid'] = f_inner_init((hidden_dim, 4 * hidden_dim))

    params['b'] = K.zeros((1, 4 * hidden_dim))

    params['W_cell'] = f_init2((3, 1, hidden_dim))

    # output hash code
    params['W_output'] = f_init_((hidden_dim, 1))
    params['b_output'] = K.zeros((1,))

    params['W_ctx_1'] = f_init_((1, 10)) # 8
    params['W_ctx_2'] = f_init_((hidden_dim, 10))
    params['W_ctx_3'] = f_init_((att_frame, 1))
    params['b_ctx'] = K.zeros((1, 10))

    params['W_feat'] = f_init_((dim_frame, steps))
    params['b_feat'] = K.zeros((steps,))

    params['W_att'] = f_init_((hidden_dim, att_frame))
    params['b_att'] = K.zeros((1, att_frame))

    params['gamma'] = K.ones((hidden_dim,))
    params['beta'] = K.zeros((hidden_dim,))

    return params


def batchnorm(X, batch_size, hidden_dim, gamma, beta, running_mean, running_std, epsilon=1e-10, axis=1, momentum=0.99, train=False):

    X = K.reshape(X, (batch_size, hidden_dim))
    input_shape = (batch_size, hidden_dim) # (1, 512)
    reduction_axes = list(range(len(input_shape))) # [0, 1]
    del reduction_axes[axis] # [0]
    broadcast_shape = [1] * len(input_shape) # [1, 1]
    broadcast_shape[axis] = input_shape[axis] # [1, 512]
    if train:
            m = K.mean(X, axis=reduction_axes) # m.shape = (1, 512), note that if matrix is 1-d then mean function will return one number even if axis=0
            brodcast_m = K.reshape(m, broadcast_shape) # m.shape = (1, 512)
            std = K.mean(K.square(X - brodcast_m) + epsilon, axis=reduction_axes) # batchnormed m(m**2)
            std = K.sqrt(std) # batchnormed m, (1, 512)
            brodcast_std = K.reshape(std, broadcast_shape) # (1, 512)
            mean_update = momentum * running_mean + (1 - momentum) * m # (1, 512)
            std_update = momentum * running_std + (1 - momentum) * std # (1, 512)
            X_normed = (X - brodcast_m) / (brodcast_std + epsilon) # (1, 512)
    else:
            brodcast_m = K.reshape(running_mean, broadcast_shape)
            brodcast_std = K.reshape(running_std, broadcast_shape)
            X_normed = ((X - brodcast_m) /
                            (brodcast_std + epsilon))
    out = K.reshape(gamma, broadcast_shape) * X_normed + K.reshape(beta, broadcast_shape) # (1, 512)

    return out, mean_update, std_update


class lstm_simple(object):
    def __init__(self, feature, attribute, options, LSTMParams):
        self.hidden_dim = options['hidden_dim']
        self.att_frame = options['att_frame']
        self.steps = options['steps']
        self.dim_frame = options['dim_frame']
        self.batch_size = options['batch_size']
        self.attribute = attribute
        self.feature = feature
        self.params = LSTMParams
        self.epsilon = options['epsilon']
        self.momentum = options['momentum']
        self.axis_bn = options['axis_bn']

        self.init = T.dot(self.feature, self.params['W_fea']) + T.reshape(T.repeat(self.params['b_fea'], self.batch_size, 0), [self.batch_size, 1, 1, 2*self.hidden_dim])
        self.init = T.cast(self.init, 'float32')
        self.init = T.reshape(self.init, [self.batch_size, 2 * self.hidden_dim])

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def step(self, cell_p, hid_p):

        embed = T.reshape(T.dot(self.attribute[:, 0], self.params['W_ctx_3']), [self.batch_size, 10])
        hidP = T.dot(hid_p, self.params['W_ctx_2']) # (25, 10)
        embedd = T.repeat(self.params['W_ctx_1'], self.batch_size, 0) * T.tanh(embed + hidP + T.repeat(self.params['b_ctx'], self.batch_size, 0)) # (25, 10)
        alpha_base = T.reshape(T.exp(embedd), [self.batch_size, 10, 1]) # (25, 10, 1)
        alpha_base = alpha_base / alpha_base.sum()
        att = T.reshape(self.attribute[:, 0], [self.batch_size, 10, self.att_frame])
        ctx = (alpha_base * att / T.reshape(alpha_base.sum(axis=1), [self.batch_size, 1, 1])).sum(axis=1) # (25, 300)
        ctx = T.reshape(ctx, [self.batch_size, self.att_frame])
        # ctx += T.dot(hid_p, self.params['W_att']) + T.repeat(self.params['b_att'], self.batch_size, 0)

        input_to = T.dot(ctx, self.params['W_in']) + T.repeat(self.params['b'], self.batch_size, 0) # (25, 2048)
        gate = input_to + T.dot(hid_p, self.params['W_hid'])

        # Apply nonlinearities
        ingate = T.nnet.sigmoid(self._slice(gate, 0, self.hidden_dim) + cell_p * T.repeat(self.params['W_cell'][0], self.batch_size, 0))
        forgetgate = T.nnet.sigmoid(self._slice(gate, 1, self.hidden_dim) + cell_p * T.repeat(self.params['W_cell'][1], self.batch_size, 0))
        cell_input = T.tanh(self._slice(gate, 2, self.hidden_dim))

        # Compute new cell value
        cell = forgetgate * cell_p + ingate * cell_input

        # BatchNormalization
        # brodcast_m = K.reshape(mean_p, broadcast_shape)
        # brodcast_std = K.reshape(std_p, broadcast_shape)
        # cell_normed = ((cell - brodcast_m) /
        #                 (brodcast_std + self.epsilon))
        broadcast_shape = [self.batch_size, self.hidden_dim]
        cell_bn = K.reshape(self.params['gamma'], broadcast_shape) * cell + K.reshape(self.params['beta'], broadcast_shape) # (1, 512)

        outgate = T.nnet.sigmoid(self._slice(gate, 3, self.hidden_dim) + cell_bn * T.repeat(self.params['W_cell'][2], self.batch_size, 0))

        # Compute new hidden unit activation
        hid = outgate * T.tanh(cell_bn)
        return T.reshape(cell_bn, [self.batch_size, self.hidden_dim]), T.reshape(hid, [self.batch_size, self.hidden_dim])

    def get_Params(self):
        if hasattr(self, 'params'):
            Params = self.params
        else:
            Params = []
        return Params

    def set_output(self):
        value, _ = theano.scan(fn=self.step,
                               # non_sequences=[self.attribute],
                               outputs_info=[self._slice(self.init, 0, self.hidden_dim),
                                             self._slice(self.init, 1, self.hidden_dim)],
                               name='lstm',
                               n_steps=self.steps)
        return value


class ComputeCode(object):

    def __init__(self, feature, options, LSTMParams, state):
        self.hidden_dim = options['hidden_dim']
        self.att_frame = options['att_frame']
        self.steps = options['steps']
        self.dim_frame = options['dim_frame']
        self.batch_size = options['batch_size']
        self.feature = feature
        self.params = LSTMParams
        self.state = state

    def set_output(self):
        output_frame = T.reshape(T.dot(self.state[1], self.params['W_output']) + self.params['b_output'], [self.batch_size, self.steps])
        # output_frame = T.reshape(T.dot(self.state[1], T.eye(self.hidden_dim)), [self.batch_size, self.steps, self.hidden_dim])
        featurepart = T.dot(T.reshape(self.feature, [self.batch_size, self.dim_frame]), self.params['W_feat']) + self.params['b_feat']
        return output_frame, featurepart


class CompAndTrain(object):

    def __init__(self, options):
        self.options = options

    
    def get_params_value(self):
        # new_params = [par.get_value() for par in self.params]
        new_params = self.params
        return new_params

    def reload_params(self, params_file):
        print 'Reloading model params'
        ff = np.load(params_file)
        new_parms = ff['params']
        for idx, par in enumerate(self.params):
            K.set_value(par, new_parms[idx])

    def CompileAndUpdate(self, Params):

        weight = self.options['weight']

        fea = T.tensor4(name='input_features', dtype=theano.config.floatX)
        att = T.tensor4(name='input_att', dtype='float32')

        LSTM = lstm_simple(fea, att, self.options, Params)
        LSTMproj = LSTM.set_output()
        LSTMC = ComputeCode(fea, self.options, Params, LSTMproj)
        frame, featurepart = LSTMC.set_output()

        self.params = LSTM.get_Params()

        steps = self.options['steps']
        # for i in range(self.options['batch_size']):
        AA = K.sigmoid(frame)

        Code = AA * featurepart

        #Code = Code / T.sqrt(T.sum(T.sqr(featurepart)))

        # 	for par_name, par_value in Params:
        #     	Params[par_name] += self.options['l2_decay'] * K.sum(K.mean(K.square(par_value.get_value()), axis=0))
        #     return Params
        # Params = Regularize(Params)

        # opt = optimizer.Adam(self.params, lr=self.options['lrate'])
        # updates = opt.get_updates(self.params, loss)

        # train_graph = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], loss, on_unused_input='warn', allow_input_downcast=True)
        # self.test_graph = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], loss, on_unused_input='warn')
        my_H_last = Code
        encoder = theano.function([fea, att], my_H_last, on_unused_input='ignore', allow_input_downcast=True)

        return fea, att, Code, encoder

'''
note: the NUS-WIDE dataset has 6 redundent images, the NUSGlove.h5 and NUSfeat.h5 each has 269642 images.
classes.h5: 1000 classes for training, the number of images each classes contain is recorded in numEachClass.npy
NUSGlove.h5: word vector matrixs, top 10
NUSfeat.h5: features
'''

def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = theano.shared(pp[kk])

    return params

def load_pparams(path, params):
    pp = np.load(path)['params'].tolist()
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s not in the archive' % kk)
        params[kk] = theano.shared(pp[kk])
    return params

def zeroMean(dataMat):      
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData,meanVal

def pca(dataMat,n):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)
    
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    lowDDataMat=newData*n_eigVect
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal
    return lowDDataMat,reconMat

def run(printFreq, #frequency of printing loss
	    reload,    #whethor or not to reload
	    loadfile,  #reloaded params if needed
	    feature,
	    attribute,
	    saveto,
	    maxepoch,
	    steps,
	    dim_frame,
	    att_frame,
	    hidden_dim,
	    batch_size,
	    lrate,
	    l2_decay,
	    saveFreq,
	    validFreq,
	    weight,
        epsilon,
        momentum,
        axis_bn):

    options = {}
    options['maxepoch'] = maxepoch
    options['steps'] = steps
    options['dim_frame'] = dim_frame
    options['att_frame'] = att_frame
    options['hidden_dim'] = hidden_dim
    options['batch_size'] = batch_size
    options['saveto'] = saveto
    options['optimizer'] = optimizer
    options['lrate'] = lrate
    options['l2_decay'] = l2_decay
    options['weight'] = weight
    options['saveFreq'] = saveFreq
    options['validFreq'] = validFreq
    options['epsilon'] = epsilon
    options['momentum'] = momentum
    options['axis_bn'] = axis_bn

    featurefile = h5.File(feature, 'r')
    attfile = h5.File(attribute, 'r')

    imageclasses = h5.File('/mnt/disk1/binyi/nus/imageclas.h5', 'r')
    id_train = np.load('/mnt/disk1/binyi/nus/truetrainid.npy', 'r')
    id_test = np.load('/mnt/disk1/binyi/nus/truetestid.npy', 'r')
    group_train = h5.File('/mnt/disk1/binyi/nus/class.h5', 'r')

    LSTMParamss = init_params(options)

    number_test = -1
    if reload:
        load_pparams(loadfile, LSTMParamss)

    model = CompAndTrain(options)
    fea, att, Code, encoder = model.CompileAndUpdate(LSTMParamss)

    alllist = list(range(2052))
    shuffle(alllist)

    codList = []
    # train_loss_his = []
    idx_list = list(np.load('/mnt/disk1/binyi/nus/truetestid.npy', 'r'))
    for k in range(len(id_test)):
        buf_num = k
        number_test += options['batch_size']

        idx_train = []
        pos_idx_train = []
        neg_idx_train = []
        bufn = idx_list[buf_num]
        idx_train.append(bufn)
        # idx_list.remove(bufn)

        # print len(idx_train), len(pos_idx_train), len(neg_idx_train)
        a_all = []
        b_all = []
        for i in range(options['batch_size']):
            a = np.array([[featurefile[idx_train[i]][:]]])
            b = np.array([attfile[idx_train[i]][:]])
            bb = np.vstack((b, b, b, b))
            bbb = np.vstack((bb, bb, bb, bb))
            bbbb = np.vstack((bbb, bbb))
            a_all.append(a)
            b_all.append(bbbb)

        a_all = np.array(a_all).astype(config.floatX)
        b_all = np.array(b_all).astype(config.floatX)
        # print a_all.shape, b_all.shape
        # print len(idx_list)

        graph = 0.

        try:
                start_time = time.time()
                # every epoch passes same feature/attribute vector (* nsteps)
                # if k == 0:
                #     Old_Param = LSTMParamss
                # else:
                #     Old_Param = model.get_params_value_theano()

                # NewLoss, _, _, _, train_graph, encoder, New_Param = model.CompileAndUpdate(Old_Param)
                # newopt = optimizer.Adam(New_Param)
                # Updates = newopt.get_updates(New_Param, NewLoss)
                # loss = f_grad_shared(a_all, b_all, a_pos_all, b_pos_all, a_neg_all, b_neg_all)
                # graph = train_graph(Thistime, updates=Updates)
                # f_update(options['lrate'])
                HCode = encoder(a_all, b_all)
                # HCode = np.reshape(HCode, [steps, hidden_dim])
                # HCode = PCA(n_components=1).fit_transform(HCode)
                # HCode = np.reshape(HCode, [1, steps])
                HC = np.asarray((HCode>=0), dtype='int')
                print HC
                # train_loss_sum = loss
                # train_loss_his.append(train_loss_sum)
                codList.append(HC)
        except KeyboardInterrupt:
            print "Test interrupted"

    sio.savemat('code/test34_2_1k_96', {'test': codList})

    number_test = -1
    codList_ = []
    alllist_ = list(range(87475))
    shuffle(alllist_)
    idx_list = list(np.load('/mnt/disk1/binyi/nus/truetrainid.npy', 'r'))
    for k in range(len(id_train)):
        buf_num = k
        number_test += options['batch_size']

        idx_train = []
        pos_idx_train = []
        neg_idx_train = []
        bufn = idx_list[buf_num]
        idx_train.append(bufn)
        # idx_list.remove(bufn)

        # print len(idx_train), len(pos_idx_train), len(neg_idx_train)
        a_all = []
        b_all = []
        for i in range(options['batch_size']):
            a = np.array([[featurefile[idx_train[i]][:]]])
            b = np.array([attfile[idx_train[i]][:]])
            bb = np.vstack((b, b, b, b))
            bbb = np.vstack((bb, bb, bb, bb))
            bbbb = np.vstack((bbb, bbb))
            a_all.append(a)
            b_all.append(bbbb)

        a_all = np.array(a_all).astype(config.floatX)
        b_all = np.array(b_all).astype(config.floatX)
        # print a_all.shape, b_all.shape
        # print len(idx_list)

        graph = 0.

        try:
                start_time = time.time()
                # every epoch passes same feature/attribute vector (* nsteps)
                # if k == 0:
                #     Old_Param = LSTMParamss
                # else:
                #     Old_Param = model.get_params_value_theano()

                # NewLoss, _, _, _, train_graph, encoder, New_Param = model.CompileAndUpdate(Old_Param)
                # newopt = optimizer.Adam(New_Param)
                # Updates = newopt.get_updates(New_Param, NewLoss)
                # loss = f_grad_shared(a_all, b_all, a_pos_all, b_pos_all, a_neg_all, b_neg_all)
                # graph = train_graph(Thistime, updates=Updates)
                # f_update(options['lrate'])
                HCode = encoder(a_all, b_all)
                # HCode = np.reshape(HCode, [steps, hidden_dim])
                # HCode = PCA(n_components=1).fit_transform(HCode)
                # HCode = np.reshape(HCode, [1, steps])
                HC = np.asarray((HCode>=0), dtype='int')
                print HC
                # train_loss_sum = loss
                # train_loss_his.append(train_loss_sum)
                codList_.append(HC)
        except KeyboardInterrupt:
            print "Test interrupted"

    sio.savemat('code/train34_2_1k_96', {'train': codList_})

run(100,
	True,
	'/mnt/disk1/binyi/fxy/L/v34_96bit/2_1049.npz',
	'/mnt/disk1/binyi/nus/NUSF.h5',
	'/mnt/disk1/binyi/nus/NUSGlove.h5',
	'/mnt/disk1/binyi/fxy/L/v17/',
	20,
	96,
	4096,
	300,
	512,
	1,
	0.001,
	0.0001,
	4000,
	5000,
	0.01,
    1e-10,
    0.99,
    1)
