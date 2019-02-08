# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


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
        self.input = None

    def forward(self, x):

        # Might we need to store something before returning?
        self.input = x
        return 1/(1 + np.exp(-x))


    def derivative(self):

        # Maybe something we need later in here...

        raise NotImplemented


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

        self.logits = x
        self.labels = y
        exponents = np.exp(self.logits)
        # following for numerical stability
        # exponents = np.exp(self.logits - np.max(self.logits))
        self.sm = exponents/(1+np.sum(exponents))
        # computing the cross entropy for entire batch
        x_entropy_loss = -np.sum(np.dot(np.log(self.sm), y))
        return x_entropy_loss

    def derivative(self):

        # self.sm might be useful here...
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
def random_normal_weight_init(d0, d1, bsz):
    return np.random.random((d0, d1, bsz))


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
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        prev_layer_size = self.input_size
        for layer_idx, layer_size in enumerate(hiddens):
            self.W.append(weight_init_fn(prev_layer_size, hiddens[layer_idx], batch_size))
            self.dW.append(np.zeros(prev_layer_size, hiddens[layer_idx]))
            self.b.append(bias_init_fn(prev_layer_size))
            self.db.append(np.zeros(prev_layer_size))
            prev_layer_size = layer_size

        self.W.append(weight_init_fn(hiddens[-1], output_size))
        self.dW.append(np.zeros(hiddens[-1], output_size))
        self.b.append(bias_init_fn(hiddens[-1]))
        self.db.append(np.zeros(hiddens[-1]))

        self.stored_activations = []

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        # activate the weight matrices
        X_curr = x
        n_hidden = len(self.W)
        for layer_idx in range(0, n_hidden):
            X_prev = X_curr
            lin_comb = np.dot(np.transpose(self.W[layer_idx]), X_prev) + self.b[layer_idx]
            X_curr = self.activations[layer_idx].forward(lin_comb)
            # saving input and activated input for gradient computation
            self.stored_activations.append((X_prev, X_curr))

        return X_curr

    def zero_grads(self):
        raise NotImplemented

    def step(self):
        raise NotImplemented

    def backward(self, labels):

        raise NotImplemented

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

    # Setup ...
    input_size = len(trainx[0])
    output_size = 19
    hiddens = [100]
    activations = [Sigmoid()]
    weight_init_fn = random_normal_weight_init
    bias_init_fn = zeros_bias_init
    criterion = SoftmaxCrossEntropy()
    lr = 0.01
    mlp = MLP(input_size, output_size, batch_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0)

    for e in range(nepochs):

        # Per epoch setup ...
        prev_b = 0
        for b in range(0, len(trainx), batch_size):
            X_in = trainx[prev_b:b]
            labels = trainy[prev_b:b]
            logits = mlp(X_in)
            loss = mlp.criterion.forward(logits, labels)
            mlp.backward(labels)
            prev_b = b

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