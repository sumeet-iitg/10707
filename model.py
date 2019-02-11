# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os
import sys
from .data_loader import load_data


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


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = None

    def forward(self, x):

        # Might we need to store something before returning?
        self.sigmoid = 1/(1 + np.exp(-x))
        return self.sigmoid


    def derivative(self):

        # Maybe something we need later in here...

        return self.sigmoid * (1 - self.sigmoid)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

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

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


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
        exponents = np.exp(self.logits).reshape(bsz, -1, 1) # bsz x label_size x 1
        # following for numerical stability
        # exponents = np.exp(self.logits - np.max(self.logits))

        # assuming batch is the 0th dimension
        self.sm = exponents/(1+np.sum(exponents, axis=1))
        self.sm = self.sm.squeeze(axis=2) # bsz x label_size x 1 --> bsz x label_size
        # cross entropy for entire batch matrix
        x_entropy_loss = -np.sum(np.dot(np.log(self.sm), y))
        return x_entropy_loss

    def derivative(self):

        # self.sm might be useful here...
        # batch is the first dim here, i.e. these are batch of row vectors
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        # if eval:
        #    # ???

        self.x = x

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...

        raise NotImplemented

    def backward(self, delta):

        raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(bsz, d0, d1):
    return np.random.random((bsz, d0, d1))


def zeros_bias_init(d, bsz):
    raise np.zeros(d, bsz)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, batch_size, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = []
        self.dW = []
        self.b = []
        self.db = []

        prev_layer_size = self.input_size
        for layer_idx, layer_size in enumerate(hiddens):
            # because we are left multiplying W with X_in and H thereafter.
            self.W.append(weight_init_fn(batch_size, hiddens[layer_idx], prev_layer_size))
            self.dW.append(np.zeros(batch_size, hiddens[layer_idx], prev_layer_size))
            self.b.append(bias_init_fn(batch_size, prev_layer_size))
            self.db.append(np.zeros(batch_size, prev_layer_size))
            prev_layer_size = layer_size

        # weights for output layer
        self.W.append(weight_init_fn(batch_size, output_size, hiddens[-1]))
        self.dW.append(np.zeros(batch_size, output_size, hiddens[-1]))
        self.b.append(bias_init_fn(batch_size, hiddens[-1]))
        self.db.append(np.zeros(batch_size, hiddens[-1]))

        self.stored_activations = []

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        # activate the weight matrices
        bsz = x.shape[0]
        X_curr = x
        n_hidden = len(self.W)

        self.stored_activations.append(X_curr)
        for layer_idx in range(0, n_hidden):
            X_prev = X_curr # bsz x nhid[l-1]
            X_prev = np.reshape(X_prev, (bsz, 1, -1)) # bsz x 1 x nhid[l-1]
            # W = bsz x nhid[l] x nhid[l-1] | X.T = bsz x nhid[l-1] x 1
            lin_comb = np.matmul(self.W[layer_idx], X_prev.transpose([0,2,1])) + self.b[layer_idx] # bsz x nhid[l] x 1
            X_curr = self.activations[layer_idx].forward(lin_comb).squeeze(axis=2) # bsz x nhid[l]
            # saving input and activated input for gradient computation
            self.stored_activations.append(X_curr)

        return X_curr

    def zero_grads(self):
        raise NotImplemented

    def step(self):

        for layer_idx in range(0, self.nlayers):
            self.W[layer_idx] -= self.lr * self.dW[layer_idx]
            self.b[layer_idx] -= self.lr * self.db[layer_idx]

    def backward(self, labels):
        bsz = labels.shape[0]
        # dL/do
        dAct = self.criterion.derivative() # [bsz x label_size]
        assert bsz == dAct.shape[0]
        dAct = np.expand_dims(dAct, axis=2) # [bsz x label_size x 1]
        # do/dW
        h = self.stored_activations[-1] # [bsz x 100]
        h = np.expand_dims(h, axis=1) # [bsz x 1 x 100]
        # dL/dW = dL/do[bsz x label_size x 1] * do/dW [bsz x 1 x 100]
        self.dW[-1] = np.matmul(dAct, h) # [bsz x label_size x 100]
        # dL/db = dL/do x I
        dOut_prev = dAct.squeeze(axis=2)
        self.db[-1] = dOut_prev

        W_prev = self.W[-1] #[bsz x nhid[l] x nhid[l-1]]

        for layer_idx in reversed(range(0,self.nlayers-1)):
            dOut_prev = np.expand_dims(dOut_prev, axis=1)
            # dL/dh = dL/do x do/dh = dl/do x W_prev = [bsz x nhid[l+1]] x [bsz x nhid[l+1] x nhid[l]]
            dAct_curr = np.matmul(dOut_prev, W_prev) # [bsz x 1 x nhid[l]]
            # dh/dq
            dAct_Out = self.activations[layer_idx].derivative() # bsz x nhid[l] x nhid[l]
            # dL/dq = dL/dh x dh/dq
            dLoss_Out = np.matmul(dAct_curr, dAct_Out) # bsz x 1 x nhid[l]
            # dq/dW
            h = self.stored_activations[layer_idx] # bsz x nhid[l-1]
            h = np.expand_dims(h, axis=1) # bsz x 1 x nhid[l-1]
            # dL/dW = dL/dq x dq/dW
            self.dW[layer_idx] = np.matmul(dLoss_Out.transpose([0,2,1]), h) # bsz x nhid[l] x nhid[l-1]
            # dL/db = dL/dq
            self.db[layer_idx] = dLoss_Out.squeeze(axis=1)
            dOut_prev = dLoss_Out
            W_prev = self.W[layer_idx]

        # update weights
        self.step()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True


    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []


    for e in range(nepochs):

        # Per epoch setup ...
        prev_b = 0
        train_loss = 0
        for b in range(0, len(trainx), batch_size):
            # inputs are row vectors batch_size x 1568
            X_in = trainx[prev_b:b]
            # labels batch_size x 19
            labels = trainy[prev_b:b]
            logits = mlp(X_in) # bsz x out_size
            train_loss += mlp.criterion.forward(logits, labels)
            mlp.backward(labels)
            prev_b = b
            print("Training loss: {}".format(float(train_loss/batch_size)))
            training_losses.append(train_loss)

        prev_b = 0
        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented

if __name__ == '__main__':
    directory_path = "C:\\Users\\SumeetSingh\\Documents\\Lectures\\10-707\\HW-Code\\split_data_problem_5_hw1"
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    trainX = []
    trainY = []
    validX = []
    validY = []
    testX = []
    testY = []
    for file in os.listdir(directory_path):
        data, labels = load_data(os.path.abspath(os.path.join(directory_path, file)))
        if "train" in file:
            trainX = data
            trainY = labels
        elif "val" in file:
            validX = data
            validY = labels
        else:
            testX = data
            testY = labels
            # Setup ...
    dset = [(trainX, trainY), (validX, validY), (testX, testY)]
    input_size = len(trainX[0])
    output_size = 19
    hiddens = [100]
    activations = [Sigmoid()]
    weight_init_fn = random_normal_weight_init
    bias_init_fn = zeros_bias_init
    criterion = SoftmaxCrossEntropy()
    lr = 0.01
    batch_size = 32
    mlp = MLP(input_size, output_size, hiddens, batch_size, activations, weight_init_fn, bias_init_fn, criterion,
              lr, momentum=0.0, num_bn_layers=0)
    get_training_stats(mlp, dset, 150, batch_size)
