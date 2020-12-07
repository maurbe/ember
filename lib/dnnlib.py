import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D, LeakyReLU, Concatenate, Flatten, Dense, concatenate)
from tensorflow.keras.models import Model
from lib.layers import (noise_injection, MinibatchStatConcatLayer)

# ---------------------------------------------------------------------------------------

init = 'glorot_normal'

# ---------------------------------------------------------------------------------------

def down_block(x, nf, down_pool=True):
    if down_pool:
        x = Conv2D(filters=nf, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(x)
        x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)
    return x


def up_block(x, nf, skips, ms_out=True):

    x = UpSampling2D(size=2, interpolation='nearest')(x)
    x = noise_injection()(x)
    x = concatenate([x, skips])

    x = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)
    if ms_out:
        o = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=init,
            activation='relu')(x)
        return x, o
    else:
        return x


def disc_block(x, nf, stride, m=None, o=None):

    if m is not None:
        x = Concatenate()([x, m])
        x = MinibatchStatConcatLayer()(x)
    if o is not None:
        x = Concatenate()([x, o])
    x = Conv2D(filters=nf, kernel_size=3, strides=stride, padding='same', kernel_initializer=init)(x)
    x = LeakyReLU(0.2)(x)
    return x

# ---------------------------------------------------------------------------------------

def generator():

    uin = Input(shape=(None, None, 1))

    u1  = down_block(x=uin, nf=8, down_pool=False)
    u2  = down_block(x=u1, nf=16)
    u3  = down_block(x=u2, nf=32)
    u4  = down_block(x=u3, nf=64)
    u5  = down_block(x=u4, nf=128)
    u6  = down_block(x=u5, nf=256)
    u7  = down_block(x=u6, nf=512)

    # Bottleneck
    bn = down_block(x=u7, nf=512)
    obn = Conv2D(filters=1, kernel_size=1, strides=1, activation='relu',
                 padding='same', kernel_initializer=init)(bn)

    # Decoder
    g7, o7 = up_block(x=bn, nf=512, skips=u7)
    g6, o6 = up_block(x=g7, nf=512, skips=u6)
    g5, o5 = up_block(x=g6, nf=256, skips=u5)
    g4, o4 = up_block(x=g5, nf=128, skips=u4)
    g3, o3 = up_block(x=g4, nf=64, skips=u3)
    g2, o2 = up_block(x=g3, nf=32, skips=u2)
    g1 = up_block(x=g2, nf=16, skips=u1, ms_out=False)

    g = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu')(g1)

    unet = Model(inputs=uin,
                 outputs=[g, o2, o3, o4, o5, o6, o7, obn], name='wgan')
    return unet


def critic(dim=512):

    c1 = Input(shape=(dim, dim, 1)) # gas
    c2 = Input(shape=(dim, dim, 1)) # dm
    cin = Concatenate()([c1, c2])

    # multi-scale inputs
    m2 = Input(shape=(dim // 2, dim // 2, 1))
    m3 = Input(shape=(dim // 4, dim // 4, 1))
    m4 = Input(shape=(dim // 8, dim // 8, 1))
    m5 = Input(shape=(dim // 16, dim // 16, 1))
    m6 = Input(shape=(dim // 32, dim // 32, 1))
    m7 = Input(shape=(dim // 64, dim // 64, 1))
    mbn = Input(shape=(dim // 128, dim // 128, 1))
    minputs = [m2, m3, m4, m5, m6, m7, mbn]

    # multi-scale inputs dm
    o2 = Input(shape=(dim // 2, dim // 2, 1))
    o3 = Input(shape=(dim // 4, dim // 4, 1))
    o4 = Input(shape=(dim // 8, dim // 8, 1))
    o5 = Input(shape=(dim // 16, dim // 16, 1))
    o6 = Input(shape=(dim // 32, dim // 32, 1))
    o7 = Input(shape=(dim // 64, dim // 64, 1))
    obn = Input(shape=(dim // 128, dim // 128, 1))
    oinputs = [o2, o3, o4, o5, o6, o7, obn]

    # disc-first block
    c = disc_block(cin, nf=32, stride=1)
    c = disc_block(c, nf=64, stride=2)

    for (i, m, o) in zip(range(7), minputs, oinputs):
        c = disc_block(c, nf=min(128 * 2**i, 1024), stride=1, m=m, o=o)
        c = disc_block(c, nf=min(128 * 2**(i+1), 1024), stride=2)

    c = Flatten()(c)
    c = Dense(1, activation='linear')(c)

    cri = Model(inputs=[c1, m2, m3, m4, m5, m6, m7, mbn,
                        c2, o2, o3, o4, o5, o6, o7, obn], 
                outputs=c, name='critic')
    return cri
