import numpy as np

class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.02):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.weights = np.zeros(n_inputs)
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = -1 if np.matmul(self.weights, input) < 0 else 1
        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for _ in range(self.max_epochs):
            indices = np.arange(len(training_inputs))
            np.random.shuffle(indices)
            # print(indices)
            # print(type(training_inputs))
            training_inputs = np.array(training_inputs)[indices]
            labels = np.array(labels)[indices]

            for i in range(len(training_inputs)):
                if labels[i]*np.matmul(self.weights, training_inputs[i]) <= 0:
                    self.weights += self.learning_rate*labels[i]*training_inputs[i]
                    # self.learning_rate *= 0.9995
        print("lr: %f" % self.learning_rate)
        return self.weights

