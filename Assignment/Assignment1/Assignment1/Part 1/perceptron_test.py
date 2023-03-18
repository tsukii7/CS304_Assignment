import random

import numpy as np
import matplotlib.pyplot as plt
import perceptron



def show_pos(class0, class1):
    x0, y0 = class0
    x1, y1 = class1
    plt.plot(x0, y0, 'x')
    plt.plot(x1, y1, 'o', color='r')
    plt.axis('equal')
    plt.title('Samples')
    plt.show()

def generate_samples():
    # samples = np.random.multivariate_normal(loc=[5,10], scale=[1,2], size=(2,100))
    mean0 = [7, 7]
    cov0 = [[2, 0], [0, 2]]  # diagonal covari
    mean1 = [3, 3]
    cov1 = [[2, 0], [0, 2]]  # diagonal covari
    class0 = np.random.multivariate_normal(mean0, cov0, 100).T
    class1 = np.random.multivariate_normal(mean1, cov1, 100).T
    show_pos(class0, class1)
    class0 = list(zip(class0[0], class0[1], np.repeat(1, 100)))
    class1 = list(zip(class1[0], class1[1], np.repeat(-1, 100)))
    samples = class0 + class1
    random.shuffle(samples)
    print(samples)

    return samples


def test_samples(pla, test):
    test_x, test_y, test_label = zip(*test)
    test_sets = list(zip(test_x, test_y, np.repeat(1, 100)))
    hit = 0
    for i in range(40):
        if pla.forward(test_sets[i]) == test_label[i]:
            hit += 1

    print("hit %d, hit rate %.3f %%" % (hit, (hit*100.0/40)))


if __name__ == '__main__':
    samples = generate_samples()
    train = samples[:160]
    test = samples[160:]

    train_x, train_y, train_label = zip(*train)
    train_sets = list(zip(train_x, train_y, np.repeat(1, 100)))

    pla = perceptron.Perceptron(n_inputs=3)
    pla.train(train_sets, labels=train_label)

    test_samples(pla, test)

    # train_x, train_y = train
