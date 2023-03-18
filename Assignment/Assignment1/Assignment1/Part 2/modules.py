import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = {}
        self.params['weight'] = np.mat(np.random.normal(0, 0.0001, (in_features, out_features)))
        self.params['bias'] = np.mat(np.zeros(out_features))
        self.grads = {}
        self.grads['weight'] = np.mat(np.zeros((in_features, out_features)))
        self.grads['bias'] = np.mat(np.zeros(out_features))

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x.copy()
        out = x @ self.params['weight'] + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.grads['weight'] = np.mat(self.x).T @ dout
        self.grads['bias'] = dout.copy()
        dx = dout @ self.params['weight'].T
        return dx


class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = x.copy()
        x[x < 0] = 0
        out = x
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dout[self.x <= 0] = 0
        # dx = self.x
        # dx[dx <= 0] = 0
        # dx[dx > 0] = 1
        # dx = np.multiply(dx, dout)
        return dout


class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        b = x.max()
        y = np.exp(x - b)
        out = y / y.sum()
        self.out = out
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.multiply((dout - np.multiply(dout, self.out)), self.out)
        return dx


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        # out = -np.sum(np.multiply(np.array(x)[0], np.log(y)))
        out = -np.sum(np.multiply(y, np.log(x)))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        # dx = - y / x
        dx = x - y
        return dx
