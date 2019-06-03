from __future__ import absolute_import
import numpy as np
import backend as K


def get_fans(shape):
    fan_in = shape[1] if len(shape) == 3 else np.prod(shape[1:])
    fan_out = shape[2] if len(shape) == 3 else shape[0]
    return fan_in, fan_out


def uniform(shape, scale=0.01, name=None):
    return K.variable(np.random.uniform(low=-scale, high=scale, size=shape),
                      name=name)

def uniform2(shape, scale=0.1, name=None):
    return K.variable(np.random.uniform(low=-scale, high=scale, size=shape),
                      name=name)


def normal(shape, scale=0.05, name=None):
    return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)


def lecun_uniform(shape, name=None):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale, name=name)


def glorot_normal(shape, name=None):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, name=name)


def glorot_uniform(shape, name=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)


def he_normal(shape, name=None):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s, name=name)


def he_uniform(shape, name=None):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, name=name)


def orthogonal(shape, scale=1.1, name=None):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return K.variable(scale * q[:shape[0], :shape[1]], name=name)


def identity(shape, scale=1, name=None):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return K.variable(scale * np.identity(shape[0]), name=name)


def zero(shape, name=None):
    return K.zeros(shape, name=name)


def one(shape, name=None):
    return K.ones(shape, name=name)


from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')

# waan4 gei3 dak1 dong13 tin1 leoi5 gun2 dik1 mun4 paai4
# waan4 lau4 zyu6 siu3 zoek6/ lei4 hoi1 dik1 san4 taai3
# dong13 tin1 zing2 go3 sing4 si5 naa5 joeng6 hing1/heng1 faai3
# jyun4 lou6 jat1 hei2 zau2 bun3 lei5 coeng4/zoeng2 gaai1

# waan4 gei3 dak1 gaai1 dang1 ziu3 ceot1 jat1 lim5 wong4
# waan4 jin4 loeng6 naa5 fan6 mei4 wan1 dik1 bin6 dong13
# zin2 jing2 dik1 nei5 leon4 gwok3/kwok3 taai3 hou2 hon3
# jing4 zyu6 ngaan5 leoi6 coi4 gam2 sai3 hon3

# mong4 diu6 tin1 dei6 fong2 fat1 jaa5 soeng2 bat1 hei2 zi6 gei2
# jing4 mei6 mong4 soeng13 joek3 hon3 maan6 tin1 wong4 jip6 jyun5 fei1
# zau6 syun3 wui jyu5 nei5 fan1/fan6 lei4 cai1 zyut6 dik1 hei3
# jiu3 kyut3 sam1 mong4 gei3 ngo5 bin6 gei3 bat1 hei2

# ming4 jat6 tin1 dei6 zi2 hung2 paa3 jing6 bat1 ceot1 zi6 gei2
# jing4 mei6 mong4 gan1 nei5 joek3 ding6 gaa2/gaa3 jyu4 mut6 jau5 sei2
# zau6 syun3 nei5 zong3 fut3 hung1 tong4 bat1 dik6 tin1 hei3
# loeng5 ban3 baan1 baak6 dou1 ho2 jing6 dak1 nei5
