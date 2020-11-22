import os,sys,time,copy,argparse

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(35)

tf.keras.backend.clear_session()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import label_binarize as binarizer

from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert len(y_pred) == len(y_true)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def a2s(array,p=3): 
    return str( ["{0:.3f}".format(x) for x in array] )[1:-1].replace("'",'') 
    
def t2s(t):
    m = int(t/60)
    s = int(t-m*60)
    return str(m)+"m"+str(s)+"s"

def next_batch(num, data):
    """
    Return a total of `num` random samples.
    """
    indices = np.arange(0, data.shape[0])
    np.random.shuffle(indices)
    indices = indices[:num]
    batch_data = np.asarray([data[i, :] for i in indices])

    return indices, batch_data
        
def archi2layer( name, param ):
#    print(name, param)
    name = name.lower()
    if name == "input":
        if type(param) == int:
            param = (param,)
        return tf.keras.layers.InputLayer(input_shape=param)
    elif name == "dense":
        return tf.keras.layers.Dense(
            units=param[0],
            kernel_initializer=param[1],
            bias_initializer=param[2],
        )
    elif name in ["act","activ","activation"]:
        return tf.keras.layers.Activation(param)
    elif name == "relu":
        return tf.keras.layers.ReLU(max_value=param)
    elif name == "lrelu":
        return tf.keras.layers.LeakyReLU(alpha=param)
    elif name in ["drop","dropout"]:
        return tf.keras.layers.Dropout(rate=param)
    elif name == "conv":
        return tf.keras.layers.Conv2D(
            filters=param[0],
            kernel_size=param[1],
            strides=param[2],
            padding=param[3], 
            )
    elif name == "deconv":
        return tf.keras.layers.Conv2DTranspose(
            filters=param[0],
            kernel_size=param[1],
            strides=param[2],
            padding=param[3], 
            )
    elif name in ["max","maxpool","max_pool"]:
        return tf.keras.layers.MaxPool2D(
            pool_size=param[0],
            strides=param[1],
            padding=param[2],
            )
    elif name in ["umax","umaxpool","umax_pool","upmax","up_max","up_sample","usample"]:
        return tf.keras.layers.UpSampling2D(
            size=param[0],
            interpolation=param[1],
            )
    elif name in ["bn","batch_norm"]:
        return tf.keras.layers.BatchNormalization()
    elif name == "flatten":
        return tf.keras.layers.Flatten()
    elif name == "reshape":
        return tf.keras.layers.Reshape( 
            param[0], 
            input_shape=param[1]
            )
    else:
        raise NotImplemented

def get_optimizer(optimizer_name, steps=100):

    optimizer_name = optimizer_name.lower().split('|')
    
    if 'sgd' in optimizer_name[0]:
        lr,mm = optimizer_name[1:3]
        lr = 10**( - float( lr ) )
        mm = float(mm)
        if 'decay' in optimizer_name:
            lr_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                                                       initial_learning_rate=lr, 
                                                       decay_steps=steps,
                                                       decay_rate=0.1,
                                                       staircase=True)
            return tf.keras.optimizers.SGD( lr_fn, mm )
        else:
            return tf.keras.optimizers.SGD( lr, mm, False)

    elif 'adam' in optimizer_name[0]:
        lr = optimizer_name[1]
        lr = 10**( - float( lr ) )
        if 'decay' in optimizer_name:
            lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                                                       initial_learning_rate=lr, 
                                                       decay_steps=steps,
                                                       end_learning_rate=1e-4,
                                                       power=0.4,
                                                       cycle=False,
                                                       )
            return tf.keras.optimizers.Adam(lr_fn , .9, .999, 1e-3, False)
        else:
            return tf.keras.optimizers.Adam( lr   , .9, .999, 1e-3, False)
        
    elif optimizer_name in ['MNIST','USPS','20NEWS','R10K','FMNIST','STL10']:
        return get_optim(optimizer_name)
    else:
        print('* Wrong Syntax: Return Adam optimizer *')
        return tf.keras.optimizers.Adam(1e-3 , .9,.999,1e-3)
