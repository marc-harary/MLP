import numpy as np
import pandas as pd
from NN import NN
from perceptron import Perceptron
from sklearn.linear_model import Perceptron as standardPerceptron


def main():
    # load data into shuffle matrix
    df = pd.read_csv("diabetes.csv")
    data = np.array(df)
    np.random.shuffle(data)

    # split data into labels and design matrix
    design = data[:, :-1].T
    labels = data[:, -1]
    labels = labels.reshape((labels.shape[0],1)).T # keep dims to 2
    n, m_tot = design.shape

    # normalize dataset
    maxs = np.amax(design, axis=1)
    mins = np.amin(design, axis=1)
    ranges = maxs - mins
    design -= mins.reshape(len(mins),1)
    design /= ranges.reshape(len(ranges),1)

    # split into test and training data
    frac_test = .8
    split_idx = int(frac_test*m_tot)
    train_design = design[:, :split_idx]
    train_labels = labels[:, :split_idx]
    test_design = design[:, split_idx:]
    test_labels = labels[:, split_idx:]

    # fit neural network
    nn = NN(ns=[n,5,1], acts=["ReLU","ReLU","sigmoid"])
    nn.fit(train_design, train_labels, alpha=1e-2, epochs = 20_000)
    test_acc = nn.evaluate(X=test_design, Y=test_labels)
    train_acc = nn.evaluate(X=train_design, Y=train_labels)
    print("Network test set accuracy: %.5f" % test_acc)
    print("Network training set accuracy: %.5f" % train_acc)
    print()

    # fit perceptron
    perc = Perceptron()
    perc.fit(X=train_design, Y=train_labels, alpha=1e-4,
        lambd=1e-2, epochs=100_000)
    test_acc = perc.acc(X=test_design, Y=test_labels)
    train_acc = perc.acc(X=train_design, Y=train_labels)
    print("Own perceptron test set accuracy: %.5f" % test_acc)
    print("Own perceptron training set accuracy: %.5f" % train_acc)
    print()

    # fit standard template perceptron from sklearn
    clf = standardPerceptron(tol=1e-3, random_state=0)
    clf.fit(train_design.T, train_labels.squeeze())
    train_acc = clf.score(train_design.T, train_labels.squeeze())
    test_acc = clf.score(test_design.T, test_labels.squeeze())
    print("Sklearn perceptron test set accuracy: %.5f" % test_acc)
    print("Sklearn perceptron training set accuracy: %.5f"%train_acc)
    print()


if __name__ == "__main__":
    main()