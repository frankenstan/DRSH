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

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * np.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def adam(lr, tparams, grads, fea, att, pos_fea, pos_att, neg_fea, neg_att, cost):
    gshared = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], cost, updates=gsup)
    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8
    updates = []
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (T.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * np.float32(0.))
        v = theano.shared(p.get_value() * np.float32(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

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
    # params['W_in_i'] = f_init_((att_frame, hidden_dim))
    # params['W_in_f'] = f_init_((att_frame, hidden_dim))
    # params['W_in_c'] = f_init_((att_frame, hidden_dim))
    # params['W_in_o'] = f_init_((att_frame, hidden_dim))

    params['W_fea'] = f_init_((dim_frame, 2 * hidden_dim))
    # params['W_fea_h'] = f_init_((dim_frame, hidden_dim))
    # params['W_fea_c'] = f_init_((dim_frame, hidden_dim))
    params['b_fea'] = K.zeros((1, 2 * hidden_dim))
    # params['b_fea_h'] = K.zeros((1, hidden_dim))
    # params['b_fea_c'] = K.zeros((1, hidden_dim))

    params['W_hid'] = f_inner_init((hidden_dim, 4 * hidden_dim))
    # params['W_hid_i'] = f_inner_init((hidden_dim, hidden_dim))
    # params['W_hid_f'] = f_inner_init((hidden_dim, hidden_dim))
    # params['W_hid_c'] = f_inner_init((hidden_dim, hidden_dim))
    # params['W_hid_o'] = f_inner_init((hidden_dim, hidden_dim))

    params['b'] = K.zeros((1, 4 * hidden_dim))
    # params['b_i'] = K.zeros((1, hidden_dim))
    # params['b_f'] = K.zeros((1, hidden_dim))
    # params['b_c'] = K.zeros((1, hidden_dim))
    # params['b_o'] = K.zeros((1, hidden_dim))

    params['W_cell'] = f_init2((3, 1, hidden_dim))
    # params['W_cell_i'] = f_init2((1, hidden_dim))
    # params['W_cell_f'] = f_init2((1, hidden_dim))
    # params['W_cell_o'] = f_init2((1, hidden_dim))

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

    params['W_de'] = f_init_((hidden_dim, hidden_dim))
    params['b_de'] = K.zeros((hidden_dim,))
    # params = [W_in, W_fea, W_hid, b_fea, b, W_cell, W_output, b_output,
    #           W_ctx_1, W_ctx_2, W_ctx_3, b_ctx, W_feat, b_feat, W_att, b_att]

    return params



class lstm_simple(object):
    def __init__(self, feature, attribute, options, LSTMParams):
        self.hidden_dim = options['hidden_dim']
        self.att_frame = options['att_frame']
        self.steps = options['steps']
        self.dim_frame = options['dim_frame']
        self.batch_size = options['batch_size']
        self.attribute = attribute
        self.feature = feature
        self.regularizerS = []
        self.params = LSTMParams
        self.epsilon = options['epsilon']
        self.momentum = options['momentum']
        self.axis_bn = options['axis_bn']

        self.init = T.dot(self.feature, self.params['W_fea']) + T.reshape(T.repeat(self.params['b_fea'], self.batch_size, 0), [self.batch_size, 1, 1, 2 * self.hidden_dim])
        self.init = T.cast(self.init, 'float32')
        self.init = T.reshape(self.init, [self.batch_size, 2 * self.hidden_dim])

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def step(self, cell_p, hid_p, mean_p, std_p):

        embed = T.reshape(T.dot(self.attribute[:, 0], self.params['W_ctx_3']), [self.batch_size, 10])
        hidP = T.dot(hid_p, self.params['W_ctx_2']) # (25, 10)
        embedd = T.repeat(self.params['W_ctx_1'], self.batch_size, 0) * T.tanh(embed + hidP + T.repeat(self.params['b_ctx'], self.batch_size, 0)) # (25, 10)
        alpha_base = T.reshape(T.exp(embedd), [self.batch_size, 10, 1]) # (25, 10, 1)
        alpha_base = alpha_base / alpha_base.sum()
        att = T.reshape(self.attribute[:, 0], [self.batch_size, 10, self.att_frame])
        ctx = (alpha_base * att / T.reshape(alpha_base.sum(axis=1), [self.batch_size, 1, 1])).sum(axis=1) # (25, 300)
        ctx = T.reshape(ctx, [self.batch_size, self.att_frame])
        # ctx += T.dot(hid_p, self.params['W_att']) + T.repeat(self.params['b_att'], self.batch_size, 0)

        input_to = T.dot(ctx, self.params['W_in']) + T.repeat(self.params['b'], self.batch_size, 0)# (25, 2048)
        # input_to_i = T.dot(ctx, self.params['W_in_i']) + T.repeat(self.params['b_i'], self.batch_size, 0)
        # input_to_f = T.dot(ctx, self.params['W_in_f']) + T.repeat(self.params['b_f'], self.batch_size, 0)
        # input_to_o = T.dot(ctx, self.params['W_in_o']) + T.repeat(self.params['b_o'], self.batch_size, 0)
        # input_to_c = T.dot(ctx, self.params['W_in_c']) + T.repeat(self.params['b_c'], self.batch_size, 0)
        gate = input_to + T.dot(hid_p, self.params['W_hid'])
        # gate_i = input_to_i + T.dot(hid_p, self.params['W_hid_i'])
        # gate_f = input_to_f + T.dot(hid_p, self.params['W_hid_f'])
        # gate_o = input_to_o + T.dot(hid_p, self.params['W_hid_o'])
        # gate_c = input_to_c + T.dot(hid_p, self.params['W_hid_c'])

        # Apply nonlinearities
        ingate = T.nnet.sigmoid(self._slice(gate, 0, self.hidden_dim) + cell_p * T.repeat(self.params['W_cell'][0], self.batch_size, 0))
        forgetgate = T.nnet.sigmoid(self._slice(gate, 1, self.hidden_dim) + cell_p * T.repeat(self.params['W_cell'][1], self.batch_size, 0))
        cell_input = T.tanh(self._slice(gate, 2, self.hidden_dim))

        # Compute new cell value
        cell = forgetgate * cell_p + ingate * cell_input

        # BatchNormalization
        input_shape = (self.batch_size, self.hidden_dim) # (1, 512)
        cell = K.reshape(cell, input_shape)
        reduction_axes = list(range(len(input_shape))) # [0, 1]
        del reduction_axes[self.axis_bn] # [0]
        broadcast_shape = [1] * len(input_shape) # [1, 1]
        broadcast_shape[self.axis_bn] = input_shape[self.axis_bn] # [1, 512]
        # m = K.mean(cell, axis=reduction_axes) # m.shape = (1, 512), note that if matrix is 1-d then mean function will return one number even if axis=0
        m = K.mean(cell, axis=0)
        brodcast_m = K.reshape(m, [1, self.hidden_dim]) # m.shape = (1, 512)
        # brodcast_m = m
        std = K.mean(K.square(cell - brodcast_m) + self.epsilon, axis=reduction_axes) # batchnormed m(m**2)
        std = K.sqrt(std) # batchnormed m, (1, 512)
        brodcast_std = K.reshape(std, broadcast_shape) # (1, 512)
        mean_update = self.momentum * mean_p + (1 - self.momentum) * m # (1, 512)
        std_update = self.momentum * std_p + (1 - self.momentum) * std # (1, 512)
        cell_normed = (cell - brodcast_m) / (brodcast_std + self.epsilon) # (1, 512)
        cell_bn = K.reshape(self.params['gamma'], broadcast_shape) * cell_normed + K.reshape(self.params['beta'], broadcast_shape) # (1, 512)

        # cell_bn, mean, std = batchnorm(cell, self.batch_size, self.hidden_dim, self.params['gamma'], self.params['beta'], mean_p, std_p, train=True)

        outgate = T.nnet.sigmoid(self._slice(gate, 3, self.hidden_dim) + cell_bn * T.repeat(self.params['W_cell'][2], self.batch_size, 0))

        # Compute new hidden unit activation
        hid = outgate * T.tanh(cell_bn)
        return T.reshape(cell_bn, [self.batch_size, self.hidden_dim]), T.reshape(hid, [self.batch_size, self.hidden_dim]), mean_update, std_update

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
                                             self._slice(self.init, 1, self.hidden_dim),
                                             K.zeros((self.hidden_dim,)),
                                             K.ones((self.hidden_dim,))],
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
        self.regularizerS = []
        # for par_name, par_value in Params:
        #     regularizer = regularizers.WeightRegularizer(l1=0., l2=self.options['l2_decay'])
        #     regularizer.set_param(par_value.get_value())
        #     self.regularizerS.append(regularizer)

        weight = self.options['weight']

        fea = T.tensor4(name='input_features', dtype=theano.config.floatX)
        att = T.tensor4(name='input_att', dtype='float32')
        pos_fea = T.tensor4(name='pos_fea', dtype='float32')
        pos_att = T.tensor4(name='pos_att', dtype='float32')
        neg_fea = T.tensor4(name='pos_fea', dtype='float32')
        neg_att = T.tensor4(name='neg_att', dtype='float32')
        TT = [fea, att, pos_fea, pos_att, neg_fea, neg_att]

        LSTM = lstm_simple(fea, att, self.options, Params)
        LSTMproj = LSTM.set_output()
        LSTMC = ComputeCode(fea, self.options, Params, LSTMproj)
        frame, featurepart = LSTMC.set_output()

        LSTM_pos = lstm_simple(pos_fea, pos_att, self.options, Params)
        LSTMproj_pos = LSTM_pos.set_output()
        LSTMC_pos = ComputeCode(pos_fea, self.options, Params, LSTMproj_pos)
        frame_pos, featurepart_pos = LSTMC_pos.set_output()

        LSTM_neg = lstm_simple(neg_fea, neg_att, self.options, Params)
        LSTMproj_neg = LSTM_neg.set_output()
        LSTMC_neg = ComputeCode(neg_fea, self.options, Params, LSTMproj_neg)
        frame_neg, featurepart_neg = LSTMC_neg.set_output()

        self.params = LSTM.get_Params()
        
        steps = self.options['steps']
        self.loss_1 = self.loss2 = self.loss_3 = 0.
        loss = 0.
        # for i in range(self.options['batch_size']):
        AA = K.sigmoid(frame)
        BB = K.sigmoid(frame_pos)
        CC = K.sigmoid(frame_neg)

        Code = AA * featurepart
        Code_pos = BB * featurepart_pos
        Code_neg = CC * featurepart_neg

        Code_ = (Code>=0).astype('float32')
        Code_pos_ = (Code_pos>=0).astype('float32')
        Code_neg_ = (Code_neg>=0).astype('float32')
        # Code = Code / T.sqrt(T.sum(T.sqr(featurepart)))
        # Code_pos = Code_pos / T.sqrt(T.sum(T.sqr(Code_pos)))
        # Code_neg = Code_neg / T.sqrt(T.sum(T.sqr(Code_neg)))

        self.loss2 = T.max((0, 2. - T.sqrt(T.sum(T.sqr(Code - Code_neg))) / 32. + T.sqrt(T.sum(T.sqr(Code - Code_pos))) / 32.))
        for i in range(32):
            self.loss_3 += T.max((0, 2. - T.sqrt(T.sum(T.sqr(Code_[0][i] - Code_neg_[0][i]))) + T.sqrt(T.sum(T.sqr(Code_[0][i] - Code_pos_[0][i])))))

        loss = self.loss2 + 0.1 * self.loss_3

        for par in Params.values():
            loss += K.sum(K.square(par)) * self.options['l2_decay'] / 2.
        # def Regularize(Params):
        #   for par_name, par_value in Params:
        #       Params[par_name] += self.options['l2_decay'] * K.sum(K.mean(K.square(par_value.get_value()), axis=0))
        #     return Params
        # Params = Regularize(Params)

        # opt = optimizer.Adam(self.params, lr=self.options['lrate'])
        # updates = opt.get_updates(self.params, loss)

        # train_graph = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], loss, on_unused_input='warn', allow_input_downcast=True)
        # self.test_graph = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], loss, on_unused_input='warn')
        my_H_last = Code
        encoder = theano.function([fea, att, pos_fea, pos_att, neg_fea, neg_att], my_H_last, on_unused_input='ignore', allow_input_downcast=True)

        return loss, fea, att, pos_fea, pos_att, neg_fea, neg_att, Code, encoder

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

    epoch_n = 0
    number_train = -1
    if reload:
        load_params(loadfile, LSTMParamss)
        number_train = 83999
        epoch_n = 0

    model = CompAndTrain(options)
    loss, fea, att, pos_fea, pos_att, neg_fea, neg_att, Code, encoder = model.CompileAndUpdate(LSTMParamss)
    grads = T.grad(loss, wrt=list(LSTMParamss.values()))
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = adam(lr, LSTMParamss, grads, fea, att, pos_fea, pos_att, neg_fea, neg_att, loss)

    alllist = list(range(87475))
    shuffle(alllist)

    train_loss_his = []
    for k in range(options['maxepoch'] * len(id_train) / options['batch_size']):
        if number_train + 1 + options['batch_size'] >= 87475:
            buf_1 = alllist[number_train + 1:]
            buf_2 = alllist[: np.mod(number_train + 1 + options['batch_size'], 10000)]
            buf_num = buf_1 + buf_2
        else:
            buf_num = alllist[number_train + 1: number_train + 1 + options['batch_size']]
        number_train += options['batch_size']
        if number_train >= 87475:
                number_train -= 87475
                epoch_n += 1
                shuffle(alllist)

        idx_train = []
        pos_idx_train = []
        neg_idx_train = []
        for n in range(options['batch_size']):
            idx_list = list(np.load('/mnt/disk1/binyi/nus/truetrainid.npy', 'r'))
            bufn = idx_list[buf_num[n]]
            idx_train.append(bufn)
            idx_list.remove(bufn)
            # for pos
            cate_train = imageclasses[bufn].keys() # category(categories) the image belongs to
            shuffle(cate_train)
            A = group_train[cate_train[0]].keys()
            A.remove(bufn)
            shuffle(A)
            for q in range(len(A)):
                if A[q] not in id_test:
                    pos_idx_train.append(A[q])
                    break
            # for neg
            B = [u'animal', u'beach', u'bridge', u'cat', u'clouds', u'flowers', u'grass', u'house', u'lake', u'mountain', u'ocean', u'reflection', u'sand', u'sky', u'snow', u'street', u'sun', u'sunset', u'tree', u'water', u'window']
            for j in range(len(cate_train)):
                B.remove(cate_train[j])
            shuffle(B)
            nega = group_train[B[0]].keys()
            shuffle(nega)
            for k in range(len(nega)):
                if nega[k] not in id_test:
                    neg_idx_train.append(nega[k])
                    break
        # print len(idx_train), len(pos_idx_train), len(neg_idx_train)
        a_all = []
        b_all = []
        a_pos_all = []
        b_pos_all = []
        a_neg_all = []
        b_neg_all = []
        for i in range(options['batch_size']):
            a = np.array([[featurefile[idx_train[i]][:]]])
            a_pos = np.array([[featurefile[pos_idx_train[i]][:]]])
            a_neg = np.array([[featurefile[neg_idx_train[i]][:]]])
            b = np.array([attfile[idx_train[i]][:]])
            b_pos = np.array([attfile[pos_idx_train[i]][:]])
            b_neg = np.array([attfile[neg_idx_train[i]][:]])
            bb = np.vstack((b, b, b, b))
            bbb = np.vstack((bb, bb, bb, bb))
            bbbb = np.vstack((bbb, bbb))
            bb_pos = np.vstack((b_pos, b_pos, b_pos, b_pos))
            bbb_pos = np.vstack((bb_pos, bb_pos, bb_pos, bb_pos))
            bbbb_pos = np.vstack((bbb_pos, bbb_pos))
            bb_neg = np.vstack((b_neg, b_neg, b_neg, b_neg))
            bbb_neg = np.vstack((bb_neg, bb_neg, bb_neg, bb_neg))
            bbbb_neg = np.vstack((bbb_neg, bbb_neg))
            a_all.append(a)
            b_all.append(bbbb)
            a_pos_all.append(a_pos)
            b_pos_all.append(bbbb_pos)
            a_neg_all.append(a_neg)
            b_neg_all.append(bbbb_neg)

        a_all = np.array(a_all).astype(config.floatX)
        b_all = np.array(b_all).astype(config.floatX)
        a_pos_all = np.array(a_pos_all).astype(config.floatX)
        b_pos_all = np.array(b_pos_all).astype(config.floatX)
        a_neg_all = np.array(a_neg_all).astype(config.floatX)
        b_neg_all = np.array(b_neg_all).astype(config.floatX)
        # print a_all.shape, b_all.shape
        # print len(idx_list)

        Thistime = [a_all, b_all, a_pos_all, b_pos_all, a_neg_all, b_neg_all]
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
                loss = f_grad_shared(a_all, b_all, a_pos_all, b_pos_all, a_neg_all, b_neg_all)
                # graph = train_graph(Thistime, updates=Updates)
                f_update(options['lrate'])
                HCode = encoder(a_all, b_all, a_pos_all, b_pos_all, a_neg_all, b_neg_all)
                train_loss_sum = loss
                train_loss_his.append(train_loss_sum)

                if np.isnan(train_loss_sum) or np.isinf(loss):
                    print 'bad cost detected: ', loss

                end_time = time.time()
                print epoch_n, number_train, train_loss_sum, ('Training took %.1fs' % (end_time - start_time))
                print np.asarray((HCode>=0), dtype='int')

                # saving
                if options['saveto'] and np.mod(number_train+1+87475*epoch_n, options['saveFreq']) == 0:
                    print 'Saving...',
                    params_to_save = model.get_params_value()
                    ppaa = unzip(params_to_save)
                    np.savez(options['saveto'] + str(epoch_n) + '_' + str(number_train) + '.npz',
                             params=ppaa,
                             train_loss_his=train_loss_his, **ppaa)
                    # pkl.dump(options, open('%s.pkl' % (options['saveto'] + '.npz'), 'wb'), -1)
                    print 'Save Done'

        except KeyboardInterrupt:
            print "Test interrupted"

run(100,
    False,
    '/mnt/disk1/binyi/fxy/L/vfff_batch/0_83999.npz',
    '/mnt/disk1/binyi/nus/NUSF.h5',
    '/mnt/disk1/binyi/nus/NUSGlove.h5',
    '/mnt/disk1/binyi/fxy/L/v34/',
    20,
    32,
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
