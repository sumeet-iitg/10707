import numpy as np
import os
import sys
from data_loader import load_data
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle

from utils import im2col_indices, col2im_indices

class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()
        self.relu= None

    def forward(self, x):
        self.relu = np.maximum(x, 0)
        return self.relu

    def derivative(self):
        return 1.0*(self.relu > 0)


def random_normal_weight_init(dimensions):
    return np.random.uniform(-1, 1, size=dimensions)


def zeros_bias_init(d):
    return np.zeros(d)

class Layer(object):
    def __init__(self):
        super(Layer, self).__init__()
        self.input = None

    def forward(self, x):
        return NotImplementedError

    def backward(self, delta):
        return NotImplementedError

class Conv2d(Layer):
    def __init__(self, input_shape, num_filters=1, kernel_size=3, stride=1, padding=0):
        super(Conv2d,self).__init__()

        self.channels, self.height_in, self.width_in = input_shape
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # weights of each filter
        self.W = random_normal_weight_init((num_filters, self.channels, kernel_size, kernel_size))
        self.dW = np.zeros(self.W.shape)

        self.b = np.zeros((num_filters,1))
        self.db = np.zeros((num_filters,1))

        self.conv_out_ht  = (self.height_in - kernel_size + 2 * self.padding)/self.stride + 1 # num of rows
        self.conv_out_wd = (self.width_in - kernel_size + 2 * self.padding)/self.stride + 1 # num of cols

    def forward(self, x):
        bsz = x.shape[0]
        input_to_cols = im2col_indices(x, self.kernel_size, self.kernel_size) # [WF*HF*C, out_wd*bsz]
        self.input = input_to_cols
        conv_out = self.W.reshape(self.num_filters, -1).dot(input_to_cols) + self.b

        conv_out = conv_out.reshape(self.num_filters, self.conv_out_ht, self.conv_out_wd, bsz)
        conv_out = conv_out.transpose(3,0,1,2)
        return conv_out

    def backward(self, delta):
        db = np.sum(delta, axis=(0,2,3)) # sum along all but the filters dimension
        self.db = db.reshape(self.num_filters, -1)

        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        self.dW = delta_reshaped.dot(self.input.T).reshape(self.W.shape)

        return self.dW, self.db

class Pooling(Layer):
    def __init__(self, input_shape, pool_size=3):
        super(Pooling,self).__init__()
        self.num_filters, self.height_in, self.width_in = input_shape
        self.pool_size = pool_size
        self.stride = pool_size
        self.height_out = (self.height_in - pool_size)/pool_size + 1
        self.width_out = (self.width_in - pool_size)/pool_size + 1


    def forward(self, x):
        self.bsz = x.shape[0]
        # reshape to bsz*Filters, 1, H, W to make im2col arrange pool strides column-wise
        x_reshaped = x.reshape(self.bsz * self.num_filters, 1, self.height_in, self.width_in)
        input_to_cols = im2col_indices(x_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)  # [H*W, out_wd*bsz*F]
        self.input = input_to_cols
        max_ids = np.argmax(input_to_cols, axis=0)
        self.max_ids = max_ids
        pool_out = input_to_cols[max_ids, :].reshape(self.num_filters, self.height_out, self.width_out, self.bsz)
        pool_out = pool_out.transpose(3, 0, 1,2)

        return pool_out

    def backward(self, delta):

        dX_cols = np.zeros(self.input.shape) # bsz,
        # delta shape: bsz, num_filters, h_out, w_out --> num_filters, h_out, w_out, bsz
        delta_flat = delta.transpose(2, 3, 0, 1).ravel()
        dX_cols[self.max_ids, :] = delta_flat

        dX = col2im_indices(dX_cols, (self.bsz * self.num_filters, 1,  self.height_in, self.width_in),
                            self.pool_size, self.pool_size, padding=0, stride=self.stride)

        return dX.reshape(self.bsz ,self.num_filters, self.height_in, self.width_in)

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        super(FullyConnectedLayer,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.W = random_normal_weight_init((output_size, input_size))
        self.dW = np.zeros(self.W.shape)
        self.b = np.zeros(output_size)
        self.db = np.zeros(self.b.shape)

    def forward(self, x):
        self.input = x
        lin_comb = np.matmul(self.W, x.T).T + self.b
        return lin_comb # batch_size, out_size

    def backward(self, delta):
        # delta = bsz x out_size
        self.dW = np.matmul(delta.T, self.input) # out_size x in_size
        self.db = np.sum(delta, axis=0)
        self.dX = np.matmul(delta, self.W) # bsz x out_size x out_size x In_size

        return self.dW, self.db, self.dX


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        bsz = x.shape[0]
        self.logits = x # bsz x label_size
        self.labels = y # bsz x label_size
        exponents = np.exp(self.logits)
        # following for numerical stability
        # exponents = np.exp(self.logits - np.max(self.logits,axis=1)
        sum_exp = np.sum(exponents, axis=1, keepdims=True)
        # assuming batch is the 0th dimension
        self.sm = exponents/(sum_exp) # bsz x label_size x 1 --> bsz x label_size

        # cross entropy for entire batch matrix
        # element-wise multiply bsz x label_size & bsz x label_size
        x_entropy_loss = -np.multiply(np.log(self.sm), y) # bsz x label_size
        return self.sm, x_entropy_loss

    def derivative(self):

        # self.sm might be useful here...
        # batch is the first dim here, i.e. these are batch of row vectors
        return self.sm - self.labels





