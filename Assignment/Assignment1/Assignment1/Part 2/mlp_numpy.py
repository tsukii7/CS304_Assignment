from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from modules import *



class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.fc = []
        self.fc.append(Linear(n_inputs, n_hidden[0]))
        for i in range(len(n_hidden) - 1):
            self.fc.append(Linear(n_hidden[i], n_hidden[i + 1]))

        self.relu = []
        for i in range(len(n_hidden)):
            self.relu.append(ReLU())

        self.out = Linear(n_hidden[-1], n_classes)
        self.softmax = SoftMax()
        self.loss = CrossEntropy()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        for i in range(len(self.n_hidden)):
            x = self.fc[i].forward(x)
            x = self.relu[i].forward(x)

        x = self.out.forward(x)
        out = self.softmax.forward(x)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dout = self.softmax.backward(dout)
        dout = self.out.backward(dout)

        grad_weight = []
        grad_bias = []

        grad_weight.append(self.out.grads['weight'])
        grad_bias.append(self.out.grads['bias'])

        for i in range(len(self.n_hidden)):
            i = len(self.n_hidden)- i - 1
            dout = self.relu[i].backward(dout)
            dout = self.fc[i].backward(dout)

        for i in range(len(self.n_hidden)):
            grad_weight.append( self.fc[i].grads['weight'])
            grad_bias.append( self.fc[i].grads['bias'])

        # for i in range(len(self.n_hidden)):
        #     self.fc[i].params['weight'] -= self.fc[i].grads['weight']
        #     self.fc[i].params['bias'] -= self.fc[i].grads['bias']

        return grad_weight, grad_bias
