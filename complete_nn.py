import numpy as np


# import torch as tr

# np.random.seed(100)


def sigmoid(t):
    # return - np.log(1. + np.exp(-t))
    return 1.0 / (1.0 + np.exp(-t))


# def sigmoid(x):
#     return np.where(x >= 0,
#                     1 / (1 + np.exp(-x)),
#                     np.exp(x) / (1 + np.exp(x)))


def softmax(p):
    return (np.exp(p).T / np.sum(np.exp(p), axis=1)).T
    

def sigmoid_derivative(p):
    return p * (1.0 - p)


def tanh(x):
    return np.tanh(x)


# take input as tanh(x):
def tanh_derivative(p):
    return 1.0 - p ** 2


def erelu(x):
    # return np.maximum(0, x)
    return np.where(x > 0, x, x * 0.55)

def derelu(x):
    dx = np.ones_like(x)
    dx[x<0] = 0.55
    # return (x > 0) * 1
    return dx

def initialize_weights(input_size, output_size):
    return np.random.uniform(-1, 1, (input_size, output_size))


"""
x shape: [batchSize, features_length]
s: input to a node
z: output from node
w['23']: is weight matrix from layer 2 to 3

num_nodes: is a list. The NeuralNetwork class form layers with different number of nodes.
num_layers: total number of hidden layers.
eta: learning rate
"""


class NeuralNetwork:
    def __init__(self, x, y, num_layers=2, num_nodes=None, eta=0.01, max_iters=100, batch_size=200, lmda=0.001):
        """

        :type num_layers: int
        :type x: np.array
        :type y: np.array

        """

        if num_nodes is None:
            num_nodes = [2, 2]
        self.y = y
        self.x = x
        self.lr = eta
        self.max_iters = max_iters
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.lmda = lmda
        self.loss = {"train": [],
                     "test": []}
        self.misclassification = {"train": [],
                                  "test": []}

        self.accuracy = {"train": [],
                         "test": []}

        self.weights = {}
        self.bias = {}
        self.s = {}
        self.z = {}
        self.delta = {}

        self.batch_size = batch_size
        self.batch_idx = 0
        self.train_batches = len(self.x["train"]) // self.batch_size
        self.test_batches = len(self.x["test"]) // self.batch_size

        # Initialize weights:
        input_shape = self.x["train"].shape[-1]
        output_shape = self.y["train"].shape[-1]
        self.weights['01'] = initialize_weights(input_shape + 1, num_nodes[0])
        for i in range(num_layers - 1):
            layer_index = str(i + 1) + str(i + 2)
            self.weights[layer_index] = initialize_weights(num_nodes[i] + 1, num_nodes[i + 1])

        out_layer_index = str(num_layers) + str(num_layers + 1)
        self.weights[out_layer_index] = initialize_weights(num_nodes[-1] + 1, output_shape)

    def split(self, batch_type):
        if self.batch_idx == self.train_batches:
            self.batch_idx = 0

        xtrain_batch = self.x[batch_type][self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
        ytrain_batch = self.y[batch_type][self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]

        self.batch_idx += 1
        return xtrain_batch, ytrain_batch

        

    def train(self):

        for ep in range(self.max_iters):
        
            for idx in range(self.train_batches):
                x_batch, y_batch = self.split("train")
                self.backprop(x_batch, y_batch)
                self.update_weights()
        
            # Loss compuation after epochs
            self.cross_entropy_loss("train")
            self.cross_entropy_loss("test")

            self.classification_errors("train")
            self.classification_errors("test")

            print("ep: {}, train_acc: {}, test_acc: {},train_loss: {}, test_loss: {}".format(ep, self.accuracy["train"][-1],
                                                                                             self.accuracy["test"][-1],
                                                                                             self.loss["train"][-1],
                                                                                             self.loss["test"][-1]))

    def predict(self, x):
        out = self.feedforward(x)
        out = (out > 0.5) * 1.0
        return out

    #  returns logits and softmax
    def feedforward(self, x):
        self.z[str(0)] = np.hstack([x, np.ones([x.shape[0], 1])])
        for i in range(self.num_layers + 1):
            x = self.z[str(i)]
            # print(x.shape)
            weight_index = str(i) + str(i + 1)
            # print(x.shape, self.weights[weight_index].shape)
            # input of layer
            self.s[str(i + 1)] = np.matmul(x, self.weights[weight_index])
            #  compute the activation of the layer
            x = erelu(self.s[str(i + 1)])
            # add bias variable
            self.z[str(i + 1)] = np.hstack([x, np.ones([x.shape[0], 1])])

        # remove bias from output
        self.z[str(self.num_layers + 1)] = self.z[str(self.num_layers + 1)][:, :-1]

        softmax_op = softmax(self.z[str(self.num_layers + 1)])
        return self.z[str(self.num_layers + 1)], softmax_op  # logits, probabilities

    def compute_delta_per_layer(self, out, y):
        layer_index = str(self.num_layers + 1)
        self.delta[layer_index] = (out - y)  # * sigmoid_derivative(self.z[layer_index])  # + self.lmda * self.s[layer_index]
        # print(self.delta[layer_index].shape)
        layer_index = str(self.num_layers)

        self.delta[layer_index] = derelu(self.z[layer_index]) * np.matmul(self.delta[str(self.num_layers + 1)],
                                                                         self.weights[
                                                                             layer_index + str(
                                                                                 self.num_layers + 1)].T)
        # print(self.delta[layer_index].shape)
        for i in range(self.num_layers - 1, 0, -1):
            layer_index = str(i)
            # print(i)
            self.delta[layer_index] = derelu(self.z[layer_index]) * np.matmul(self.delta[str(i + 1)][:, :-1],
                                                                             self.weights[
                                                                                 layer_index + str(
                                                                                     i + 1)].T)  # + self.lmda * self.s[layer_index]
            # print(self.delta[layer_index].shape)
            # print('\n')

            # + self.lmda * self.s[layer_index]

    def update_weights(self):
        del_last = self.delta[str(self.num_layers + 1)]
        # This is done to compute all updates in one layer
        self.delta[str(self.num_layers + 1)] = np.hstack([del_last, np.ones([del_last.shape[0], 1])])
        for i in range(self.num_layers, -1, -1):
            weight_index = str(i) + str(i + 1)
            weight_change_factor = self.lr * np.matmul(self.z[str(i)].T, self.delta[str(i + 1)][:, :-1]) + self.lmda * \
                                    self.weights[weight_index]
            self.weights[weight_index] = self.weights[weight_index] - weight_change_factor / self.batch_size

    def backprop(self, x, y):
        logits, prob = self.feedforward(x)
        self.compute_delta_per_layer(prob, y)

    def cross_entropy_loss(self, batch_type="train"):

        total_batches = self.train_batches if batch_type == "train" else self.test_batches
        loss = 0
        for i in range(total_batches):
            x = self.x[batch_type][i * self.batch_size: (i + 1) * self.batch_size]
            y = self.y[batch_type][i * self.batch_size: (i + 1) * self.batch_size]
            _, out = self.feedforward(x)
            loss += np.sum(-y * np.log(out + 1e-9))

        loss = loss / len(self.x[batch_type])

        self.loss[batch_type].append(loss)
        # loss = cross
        # return loss

    def classification_errors(self, batch_type="train"):
        total_batches = self.train_batches if batch_type == "train" else self.test_batches
        misclassification = 0
        for i in range(total_batches):
            x = self.x[batch_type][i * self.batch_size: (i + 1) * self.batch_size]
            y = self.y[batch_type][i * self.batch_size: (i + 1) * self.batch_size]
            _, out = self.feedforward(x)

            y_argmax = np.argmax(y, axis=1)
            out_argmax = np.argmax(out, axis=1)

            for idx in range(len(y_argmax)):
                if y_argmax[idx] - out_argmax[idx] != 0.0:
                    misclassification += 1

        self.misclassification[batch_type].append(misclassification)
        self.accuracy[batch_type].append((len(self.x[batch_type]) - misclassification) / len(self.x[batch_type]))
        # return misclassification
