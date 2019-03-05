import numpy as np
import os
import sys
from data_loader import load_data
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle

from utils import im2col_indices, col2im_indices, toOneHot

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
    def __init__(self, input_shape, num_filters=1, kernel_size=3, stride=1, padding=1):
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
        self.bsz = x.shape[0]
        input_to_cols = im2col_indices(x, self.kernel_size, self.kernel_size) # [WF*HF*C, out_wd*bsz]
        self.input = input_to_cols
        conv_out = self.W.reshape(self.num_filters, -1).dot(input_to_cols) + self.b

        conv_out = conv_out.reshape(self.num_filters, self.conv_out_ht, self.conv_out_wd, self.bsz)
        conv_out = conv_out.transpose(3,0,1,2)
        return conv_out

    def backward(self, delta):
        db = np.sum(delta, axis=(0,2,3)) # sum along all but the filters dimension
        db = db.reshape(self.num_filters, -1)
        self.db = db/self.bsz

        delta_reshaped = delta.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        dW = delta_reshaped.dot(self.input.T).reshape(self.W.shape)
        self.dW = dW/self.bsz

        return dW, db

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
        pool_out = pool_out.transpose(3, 0, 1, 2)

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
        self.bsz = x.shape[0]
        self.input = x
        lin_comb = np.matmul(self.W, x.T).T + self.b
        return lin_comb # batch_size, out_size

    def backward(self, delta):
        # delta = bsz x out_size
        dW = np.matmul(delta.T, self.input) # out_size x in_size
        self.dW = dW/self.bsz # averaging the gradients here for neater self update
        db = np.sum(delta, axis=0)
        self.db = db/self.bsz # averaging the gradients here for neater self update
        dX = np.matmul(delta, self.W) # bsz x out_size x out_size x In_size

        return dW, db, dX


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

def forward_pass(conv_net_layers, input, labels):
    bsz = input.shape[0]
    conv_layer, relu, pooling, fully_conn_layer, sfmax_layer = conv_net_layers
    conv_out = conv_layer.forward(input)
    relu_out = relu.forward(conv_out)
    pool_out = pooling.forward(relu_out)
    # flattening the pooling output
    linear_out = fully_conn_layer.forward(pool_out.reshape(bsz, -1))
    return sfmax_layer.forward(linear_out, labels)

def backward_pass(conv_net_layers):
    conv_layer, relu, pooling, fully_conn_layer, sfmax_layer = conv_net_layers

    prev_dW2 = fully_conn_layer.dW
    prev_db2 = fully_conn_layer.db
    prev_dW1 = conv_layer.dW
    prev_db1 = conv_layer.db

    # dL/dO
    dLoss_final = sfmax_layer.derivative()
    dW2, db2, dX = fully_conn_layer.backward(dLoss_final)
    dX = dX.reshape(-1, pooling.num_filters, pooling.height_out, pooling.width_out)
    dLoss_pool = pooling.backward(dX)
    dLoss_relu = relu.derivative()
    dLoss_conv = np.multiply(dLoss_pool, dLoss_relu)
    dW1, db1 = conv_layer.backward(dLoss_conv)

    return [(dW1, db1, prev_dW1, prev_db1), (dW2, db2, prev_dW2, prev_db2)]

def update_pass(conv_net_layers, weight_updates, params):
    conv_layer, relu, pooling, fully_conn_layer, sfmax_layer = conv_net_layers
    dW1, db1, prev_dW1, prev_db1 = weight_updates[0]
    dW2, db2, prev_dW2, prev_db2 = weight_updates[1]

    conv_layer.W -= params.lr * (dW1 + args.momentum*prev_dW1 + 0.5*args.l2*conv_layer.W)
    conv_layer.b -= params.lr * (db1 + args.momentum*prev_db1)

    pooling.W -= params.lr * (dW2 + args.momentum * prev_dW2 + 0.5 * args.l2 * pooling.W)
    pooling.b -= params.lr * (db2 + args.momentum * prev_db2)

def train_convnet(data, params):
    train, val, test = data
    input_shape = (3, 32, 32)
    output_classes = 10
    conv_layer = Conv2d(input_shape, num_filters=params.num_Filters, kernel_size=params.kernel_size, stride=1, padding=params.pad_size)
    relu  = ReLU()
    conv_out_shape = (params.num_Filters, conv_layer.conv_out_ht, conv_layer.conv_out_wd)
    pool_layer = Pooling(conv_out_shape, params.pad_size)
    fully_connected_input_size = params.num_Filters*pool_layer.height_out*pool_layer.width_out
    full_connected_layer = FullyConnectedLayer(fully_connected_input_size, output_classes)
    sfmax_layer = SoftmaxCrossEntropy()
    conv_net_layers = (conv_layer, relu, pool_layer, full_connected_layer, sfmax_layer)

    for e in range(params.epochs):
        # train
        train_loss = 0
        for b in range(0, len(train), params.bsz):
            sfmax, loss_mat = forward_pass(conv_net_layers, train['data'][b:b+params.bsz,:], train['labels'][b:b+params.bsz,:])
            loss = np.sum(loss_mat)
            train_loss += loss
            weight_updates = backward_pass(conv_net_layers)
            update_pass(conv_net_layers, weight_updates, params)

def img_data_from_dict(data_dict, num_pts):
    # each row in this image array is an array of dim 3 x 32 x 32
    images = np.empty((num_pts, 3, 32, 32))

    for i in range(num_pts):
        img_channels = np.empty((3, 32, 32))
        img_start = 0
        for c in range(3):
            img_end = img_start + 32*32
            img_channels[c] = data_dict[i,img_start:img_end].reshape(32,32)
            img_start = img_end
        images[i] = img_channels
    return images

def get_CIFAR10_data(file_path, save_pickle_path):
    with open(file_path , 'rb') as f:
        data = pickle.load(f)

    img_data = {'train':{}, 'val':{}, 'test':{}}
    for data_type in ['train', 'val', 'test']:
        img_data[data_type]['data'] = img_data_from_dict(data[data_type]['data'], len(data[data_type]['labels']))
        img_data[data_type]['labels'] = toOneHot(data[data_type]['labels'], 10)

    with open(save_pickle_path, 'wb') as fout:
        pickle.dump(img_data, fout)

    return img_data['train'], img_data['val'], img_data['test']

def load_CIFAR10_data(image_path):
    with open(image_path , 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['val'], data['test']

if __name__ == '__main__':
    file_path = "./sampledCIFAR10/sampledCIFAR10"
    parser = argparse.ArgumentParser()

    parser.add_argument('-bsz', type=float, help='Batch Size', dest='bsz', default=32)
    parser.add_argument('-epochs', type=int, help='Num Epochs', dest='epochs', default=150)
    parser.add_argument('-l', nargs='+', type=int, help='Hidden layer sizes', dest='hidden', default=[100])
    parser.add_argument('-lr', type=float, help='Learning Rate', dest='lr', default=0.1)
    parser.add_argument('-l2', type=float, help='Regularization', dest='l2', default=0.001)
    parser.add_argument('-m', type=float, help='Momentum', dest='momentum', default=0.0)
    parser.add_argument('-d', type=float, help='Dropout', dest='dropout', default=0.0)
    parser.add_argument('-k', type=float, help='Kernel Size', dest='kernel_size', default=3)
    parser.add_argument('-nF', type=float, help='Num Filters', dest='num_Filters', default=3)
    parser.add_argument('-pd', type=float, help='Padding Size', dest='pad_size', default=1)
    parser.add_argument('-pool', type=float, help='Pool Size', dest='pool_size', default=3)
    parser.add_argument('--loss_plot', dest='loss_plot', default=False, action='store_true')
    parser.add_argument('--lr_plot', dest='lr_plot', default=False, action='store_true')
    parser.add_argument('--m_plot', dest='m_plot', default=False, action='store_true')
    parser.add_argument('--hid_plot', dest='hid_plot', default=False, action='store_true')
    parser.add_argument('--p_load', dest='p_load', default=False, action='store_true')
    # parser.add_argument('--save_model', dest='save', default=False, action='store_true')

    args = parser.parse_args()

    if len(sys.argv) > 1:
        directory_path = sys.argv[1]

    pickle_path = file_path+'_pickle'
    if not args.p_load:
        train, valid, test = get_CIFAR10_data(file_path, pickle_path)
    else:
        train, valid, test = load_CIFAR10_data(pickle_path)

    train_size = train['data'].shape[0]
    val_size = valid['data'].shape[0]
    test_size = test['data'].shape[0]
    print("sizes {} {} {}".format(train_size, val_size, test_size))
    train_convnet((train, valid, test), args)
    # output_size = 19
    # hiddens = args.hidden
    # activations = [Sigmoid() for _ in hiddens]
    #
    # weight_init_fn = random_normal_weight_init
    # bias_init_fn = zeros_bias_init
    # criterion = SoftmaxCrossEntropy()
    # lr = args.lr
    # batch_size = args.bsz
    # epoch_axis = [ep for ep in range(args.epochs)]
    #
    # mlp = MLP(input_size, output_size, hiddens, batch_size, activations, weight_init_fn, bias_init_fn, criterion,
    #               lr=args.lr, l2=args.l2, momentum=args.momentum, dropout=args.dropout, batch_norm= args.batch_norm)







