from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random

import numpy as np
from sklearn.datasets import make_moons

from mlp_numpy import MLP
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
# LEARNING_RATE_DEFAULT = 1e-2
LEARNING_RATE_DEFAULT = 1e1
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def generate_samples():
    samples = make_moons(n_samples=1000)
    samples = list(zip(samples[0], samples[1]))
    random.shuffle(samples)
    return samples


def get_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predict = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    accuracy = 100*float(np.sum(predict == true_labels)) / len(predictions)
    # accuracy = 100 - np.count_nonzero(predict - targets) * 100.0 / len(targets)
    # accuracy = 100.0 * np.count_nonzero(predict - targets) / 200
    # accuracy[epoch] = 1 - np.count_nonzero(predict - test_labels) * 1.0 / 1000
    return accuracy


def show_pos(samples, labels):
    class0, class1 = [], []
    for i in range(len(labels)):
        if labels[i] == 1:
            class1.append(samples[i])
        else:
            class0.append(samples[i])
    x0, y0 = list(zip(*class0))
    x1, y1 = list(zip(*class1))
    plt.plot(x0, y0, 'x')
    plt.plot(x1, y1, 'o', color='r')
    plt.axis('equal')
    plt.title('Samples')
    plt.show()


def train(mlp, FLAGS, unparsed):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    samples, labels = zip(*generate_samples())

    show_pos(samples, labels)

    encode_label = np.zeros((1000, 2))
    for i in range(0, 1000):
        idx = labels[i]
        encode_label[i][idx] = 1

    train_samples = samples[:800]
    train_labels = encode_label[:800]
    test_samples = samples[800:]
    test_labels = encode_label[800:]

    show_pos(train_samples, labels[:800])
    show_pos(test_samples, labels[800:])

    for j in range(FLAGS.max_steps):
        # shuffle
        total_loss = 0
        total_grad_weight = []
        total_grad_bias = []
        indices = np.arange(len(train_samples))
        np.random.shuffle(indices)
        train_samples = np.array(train_samples)[indices]
        train_labels = np.array(train_labels)[indices]

        for i in range(800):
            output = mlp.forward(train_samples[i])
            loss = mlp.loss.forward(output, train_labels[i])
            dout = mlp.loss.backward(output, train_labels[i])
            grad_weight, grad_bias = mlp.backward(dout)

            total_loss += loss
            if len(total_grad_bias) > 0:
                for k in range(len(grad_weight)):
                    total_grad_weight[k] += grad_weight[k]
                    total_grad_bias[k] += grad_bias[k]
            else:
                total_grad_weight = grad_weight
                total_grad_bias = grad_bias

        for k in range(len(total_grad_weight)):
            total_grad_weight[k] /= 800.0
            total_grad_bias[k] /= 800.0

        mlp.out.params['weight'] -= FLAGS.learning_rate * total_grad_weight[0]
        mlp.out.params['bias'] -= FLAGS.learning_rate * total_grad_bias[0]
        for i in range(0, len(total_grad_weight) - 1):
            mlp.fc[i].params['weight'] -= FLAGS.learning_rate * total_grad_weight[i + 1]
            mlp.fc[i].params['bias'] -= FLAGS.learning_rate * total_grad_bias[i + 1]
            # print("Update parameters")

        if j % FLAGS.eval_freq == 0:
            test_output = []
            print("[epoch %d] average loss: %e \t average weight gradient: %e \t average bias gradient: %e " % (
                j, (total_loss / 200.0), (total_grad_weight[1].sum()), (total_grad_bias[1].sum())))

            for i in range(800):
                test_output.append(np.array(mlp.forward(train_samples[i]))[0])
            train_acc = get_accuracy(test_output, train_labels)

            test_output = []
            for i in range(200):
                test_output.append(np.array(mlp.forward(test_samples[i]))[0])
            test_acc = get_accuracy(test_output, test_labels)
            print('Train Set Accuracy: %.2f%%\t\tTest Set Accuracy: %.2f%%\n' % (train_acc, test_acc))


def main(FLAGS, unparsed):
    """
    Main function
    """
    hidden_list = list(map(int, FLAGS.dnn_hidden_units.split(',')))
    mlp = MLP(n_inputs=2, n_hidden=hidden_list, n_classes=2)
    train(mlp, FLAGS, unparsed)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS, unparsed)
