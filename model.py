# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os
import sys
from data_loader import load_data


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
        self.relu= None

    def forward(self, x):
        self.relu = np.maximum(x, 0)
        return self.relu

    def derivative(self):
        return 1.0*(self.relu > 0)

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

def toOneHot(y, num_classes=19):
    y_hot = np.eye(num_classes)[y,:]
    return y_hot

class BatchNorm(object):

    def __init__(self, fan_in, lr, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean_set = False
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        if eval:
            norm = (x - self.running_mean)/np.sqrt(self.running_var + self.eps)
            out = self.gamma*norm + self.beta
            return out

        self.x = x # bsz x fan_in

        self.mean = np.mean(x, axis=0, keepdims=True) # 1 x fan_in
        self.var = np.var(x, axis=0, keepdims=True) # 1 x fan_in
        # bsz x fan_in
        self.x_hat = (self.x - self.mean) / np.sqrt(self.var + self.eps)
        # bsz x fan_in
        self.out = self.gamma * self.x_hat + self.beta

        if self.running_mean_set:
            self.running_mean = self.alpha*self.running_mean + (1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var + (1-self.alpha)*self.var
        else:
            self.running_mean_set = True
            self.running_mean = self.mean
            self.running_var = self.var
        return self.out

    def backward(self, delta):
        bsz = delta.shape[0]

        dx_hat = delta*self.gamma # bsz x fan_in
        var_inv = np.sqrt(self.var + self.eps) # 1 x fan_in
        x_mu = (self.x - self.mean)  # bsz x fan_in
        dvar = np.sum(dx_hat*x_mu, axis=0, keepdims=True)*-0.5*var_inv**3 # 1 x fan_in
        dmean = np.sum(dx_hat*-1*var_inv, axis=0, keepdims=True) + dvar*np.mean(-2*x_mu) # 1 x fan_in
        dx = dx_hat*var_inv + dvar*2*x_mu/bsz + dmean/bsz
        dgamma = np.sum(delta*self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(delta, axis=0, keepdims=True)
        self.gamma -= lr*dgamma
        self.beta -= lr*dbeta
        return dx

# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    # return 0.5*np.random.uniform(-1, 1, size=(d0, d1))
    b = np.sqrt(6)/np.sqrt(d0 + d1)
    return np.random.uniform(-b,b, size=(d0, d1))

def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, batch_size, activations, weight_init_fn, bias_init_fn, criterion, lr, l2, momentum=0.0, dropout=0.0, num_bn_layers=0):

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
        self.l2 = l2
        self.momentum = momentum
        self.dropout = dropout
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = []
        self.dW = []
        self.b = []
        self.db = []
        self.batch_size = batch_size

        # previous grads for momentum
        self.prev_dW = []
        self.prev_db = []
        self.dropLayers = []

        prev_layer_size = self.input_size
        layer_dims = hiddens + [output_size]
        for layer_idx, layer_size in enumerate(layer_dims):
            # because we are left multiplying W with X_in and H thereafter.
            self.W.append(weight_init_fn(layer_dims[layer_idx], prev_layer_size))
            self.dW.append(np.zeros((layer_dims[layer_idx], prev_layer_size)))
            self.prev_dW.append(np.zeros((layer_dims[layer_idx], prev_layer_size)))
            self.b.append(bias_init_fn(layer_dims[layer_idx]))
            self.db.append(np.zeros(layer_dims[layer_idx]))
            self.prev_db.append(np.zeros(layer_dims[layer_idx]))
            prev_layer_size = layer_size

        self.stored_activations = []

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = []
            for layer_idx, layer_size in enumerate(hiddens):
                self.bn_layers.append(BatchNorm(layer_size,self.lr))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        # activate the weight matrices
        bsz = x.shape[0]
        X_curr = x

        self.stored_activations = []
        self.dropLayers = []
        self.stored_activations.append(X_curr)

        for layer_idx in range(0, self.nlayers-1):
            X_prev = X_curr # bsz x nhid[l-1]
            # X_prev = np.reshape(X_prev, (bsz, 1, -1)) # bsz x 1 x nhid[l-1]
            # W = nhid[l] x nhid[l-1] | X.T = nhid[l-1] x bsz
            if self.dropout > 0 and self.train_mode:
                drop = np.random.binomial(1, self.dropout, size=X_prev.shape)/self.dropout
                X_prev *= drop
                self.dropLayers.append(drop)
            lin_comb = np.matmul(self.W[layer_idx], X_prev.T).T + self.b[layer_idx] # bsz x nhid[l]
            if self.bn:
                lin_comb = self.bn_layers[layer_idx].forward(lin_comb, not self.train_mode)
            X_curr = self.activations[layer_idx].forward(lin_comb) # bsz x nhid[l]
            # saving input and activated input for gradient computation
            self.stored_activations.append(X_curr)
        # output layer
        # (out_size, nhid[l-1]) x (nhid[l-1], bsz)
        X_curr = np.matmul(self.W[-1], X_curr.T).T + self.b[-1]  # bsz x out_size

        return X_curr

    def zero_grads(self):
        for layer_idx in range(0, self.nlayers):
            self.prev_dW[layer_idx].fill(0)
            self.prev_db[layer_idx].fill(0)
            self.dW[layer_idx].fill(0)
            self.db[layer_idx].fill(0)

    def step(self):

        for layer_idx in range(0, self.nlayers):
            self.W[layer_idx] -= self.lr * (self.dW[layer_idx] + self.momentum*(self.prev_dW[layer_idx]) + 0.5*self.l2*self.W[layer_idx])
            self.b[layer_idx] -= self.lr * (self.db[layer_idx] + self.momentum*(self.prev_db[layer_idx]))
        self.prev_dW = self.dW
        self.prev_db = self.db

    def backward(self, labels):
        bsz = labels.shape[0]
        self.prev_dW = self.dW
        self.prev_db = self.db
        # dL/do
        dAct = self.criterion.derivative() # [bsz x label_size]
        assert bsz == dAct.shape[0]
        # dAct = np.expand_dims(dAct, axis=2) # [bsz x label_size x 1]
        # do/dW
        h = self.stored_activations[-1] # [bsz x 100]
        # h = np.expand_dims(h, axis=1) # [bsz x 1 x 100]
        # dL/dW = dL/do[bsz x label_size x 1] * do/dW [bsz x 1 x 100]
        # grads = np.matmul(dAct, h) # [bsz x label_size x 100]
        # self.dW[-1] = np.sum(grads, axis=0) # sum along batch dim
        # dL/dW = dL/do[bsz x label_size] * do/dW [bsz x 100] = [label_size * 100]
        self.dW[-1] = np.matmul(dAct.T, h)/float(bsz)

        # dL/db = dL/do
        # dOut_prev = dAct.squeeze(axis=2)
        dOut_prev = dAct
        self.db[-1] = np.sum(dOut_prev, axis=0)/float(bsz)

        W_prev = self.W[-1] #[nhid[l] x nhid[l-1]]

        for layer_idx in reversed(range(0,self.nlayers-1)):
            # dL/dh = dL/do x do/dh = dl/do x W_prev = [bsz x nhid[l+1]] x [nhid[l+1] x nhid[l]]
            # ==> [[bsz , nhid[l+1]] x [nhid[l+1] , nhid[l]]]
            dAct_curr = np.matmul(dOut_prev, W_prev) # [bsz x nhid[l]]
            # dh/dq
            dAct_Out = self.activations[layer_idx].derivative() # bsz x nhid[l]
            if self.bn:
                dAct_Out = self.bn_layers[layer_idx].backward(dAct_Out)
            # dL/dq = dL/dh x dh/dq
            dLoss_Out = np.multiply(dAct_curr, dAct_Out) # bsz x nhid[l]
            # dLoss_Out = np.expand_dims(dLoss_Out, axis=1) # bsz x 1 x nhid[l]
            # dq/dW
            h = self.stored_activations[layer_idx] # bsz x nhid[l-1]
            # h = np.expand_dims(h, axis=1) # bsz x 1 x nhid[l-1]
            # dL/dW = dL/dq x dq/dW
            # grads = np.matmul(dLoss_Out.transpose([0,2,1]), h) # bsz x nhid[l] x nhid[l-1]
            # self.dW[layer_idx] = np.sum(grads, axis=0)
            if self.dropout > 0:
                h *= self.dropLayers[layer_idx]
            # bsz x nhid[l-1]
            self.dW[layer_idx] = np.matmul(dLoss_Out.T, h)/float(bsz) # nhid[l] x nhid[l-1]
            # dL/db = dL/dq

            # self.db[layer_idx] = np.sum(dLoss_Out.squeeze(axis=1), axis=0)
            self.db[layer_idx] = np.sum(dLoss_Out, axis=0)/float(bsz)
            dOut_prev = dLoss_Out
            W_prev = self.W[layer_idx]

        # update weights
        self.step()

        return self.dW

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True


    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy, trainy_num = train
    valx, valy, valy_num = val
    testx, testy, testy_num = test
    # print(len(trainx),len(trainy))

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    for e in range(nepochs):

        # Per epoch setup ...
        train_loss = 0
        grad_check = False
        num_batches = 0
        mlp.train()
        mlp.zero_grads()
        for b in range(0, len(trainx), batch_size):
            # inputs are row vectors batch_size x 1568
            X_in = trainx[b:b+batch_size]
            # labels batch_size x 19
            labels = trainy[b:b+batch_size]
            loss1 = 0
            loss2 = 0
            eps = 1e-10
            if grad_check:
                mlp.W[0][0][2] -= eps
                logits = mlp(X_in)
                _, loss1 = mlp.criterion.forward(logits, labels)
                loss1 = np.sum(loss1)
                mlp.W[0][0][2] += 2*eps
                logits = mlp(X_in)
                _, loss2 = mlp.criterion.forward(logits, labels)
                loss2 = np.sum(loss2)
                mlp.W[0][0][2] += eps
            logits = mlp(X_in) # bsz x out_size
            sfmax, loss_mat = mlp.criterion.forward(logits, labels)
            loss = np.sum(loss_mat)
            train_loss += loss
            dw = mlp.backward(labels)
            if grad_check:
                fwd_grad_w = (loss2 - loss1)/(2*eps)
                print("Gradient check:Fwd {}, Bck {}".format(fwd_grad_w, dw[0][0]))

        mlp.eval()
        logits = mlp.forward(trainx)
        sfmax, loss = mlp.criterion.forward(logits, trainy)
        train_loss = np.sum(loss)
        y_hat = np.argmax(sfmax, axis=1)
        # print(y_hat, trainy_num)
        error_rate = np.sum([y_hat[i] != trainy_num[i] for i in range(0, len(trainy_num))])/len(trainy_num)
        # error_rate= np.sum(y_hot != trainy)/len(trainy)
        print("Epoch {} Training loss:{} Train-Err:{}".format(e, train_loss/len(trainx), error_rate*100))
        training_losses.append(train_loss/len(trainx))

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

def load_save_data(directory_path):
    trainX = []
    trainY = []
    validX = []
    validY = []
    testX = []
    testY = []
    trainY_numbers = []
    validY_numbers = []
    testY_numbers = []
    for file in os.listdir(directory_path):
        data, one_hot_labels, number_labels = load_data(os.path.abspath(os.path.join(directory_path, file)))
        data = np.array(data, dtype=np.float64)
        labels = np.array(one_hot_labels)
        number_labels = np.array(number_labels)
        if "train" in file:
            trainX = data
            trainY = labels
            trainY_numbers = number_labels
            np.save("./trainX.npy", trainX)
            np.save("./trainY.npy", trainY)
            np.save("./trainY_numbers.npy", number_labels)
        elif "val" in file:
            validX = data
            validY = labels
            validY_numbers = number_labels
            np.save("./validX.npy", validX)
            np.save("./validY.npy", validY)
            np.save("./validY_numbers.npy", number_labels)
        else:
            testX = data
            testY = labels
            testY_numbers = number_labels
            np.save("./testX.npy", testX)
            np.save("./testY.npy", testY)
            np.save("./testY_numbers.npy", number_labels)

if __name__ == '__main__':
    directory_path = "C:\\Users\\SumeetSingh\\Documents\\Lectures\\10-707\\HW-Code\\split_data_problem_5_hw1"

    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    # load_save_data(directory_path)
    # Setup ...
    trainX = np.load("./trainX.npy")
    trainY = np.load("./trainY.npy")
    trainY_numbers = np.load("./trainY_numbers.npy")
    validX = np.load("./validX.npy")
    validY = np.load("./validY.npy")
    validY_numbers = np.load("./validY_numbers.npy")
    testX = np.load("./testX.npy")
    testY = np.load("./testY.npy")
    testY_numbers = np.load("./testY_numbers.npy")
    dset = [(trainX, trainY, trainY_numbers), (validX, validY, validY_numbers), (testX, testY, testY_numbers)]
    input_size = len(trainX[0])
    output_size = 19
    hiddens = [100]
    activations = [Sigmoid(), Sigmoid()]
    weight_init_fn = random_normal_weight_init
    bias_init_fn = zeros_bias_init
    criterion = SoftmaxCrossEntropy()
    lr = 0.1
    batch_size = 32
    mlp = MLP(input_size, output_size, hiddens, batch_size, activations, weight_init_fn, bias_init_fn, criterion,
              lr, l2=0.001, momentum=0, dropout=0.8, num_bn_layers=0)
    get_training_stats(mlp, dset, 150, batch_size)
