# Edward Zhao
#
# Implemented the Perceptron classification algorithm and evaluated it on Titanic survivor dataset.
# Measures the performance on the test data using hinge loss.

import pandas as pd
import numpy as np


def sign(a):
    if a >= 0:
        return 1
    else:
        return -1


def predict(row1, weights1, bias1):
    return sign(np.dot(row1, weights1) + bias1)


def accuracy(predictions, actual):
    a = 0
    total = 0
    for i in range(len(predictions)):
        if predictions[i] == list(actual)[i]:
            a += 1
        total += 1
    return a / total


def hinge_loss(row1, weights1, bias1, label):
    return max(0, 1 - label['survived'], np.dot(row1, weights1) + bias1)


def perceptron(train_data, train_label):
    df_comb = pd.concat([train_data, train_label], axis=1)
    df_comb['survived'] = df_comb['survived'].replace({0:-1})
    df_comb = df_comb.sample(frac=1)
    weights = [0] * (len(df_comb.columns) - 1)
    epochs = 100
    bias = 0
    lr = 0.01

    for epoch in range(epochs):
        c = False
        for i in range(len(df_comb)):
            row = df_comb.iloc[i]
            a = row['survived'] * sign(np.dot(row[:-1], weights) + bias)
            if a <= 0:
                c = True
                weights += row['survived'] * lr * row[:-1]
                bias += row['survived'] * lr
        if not c:
            break

    return weights, bias


if __name__ == "__main__":
    train_data = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\HW3-CS373\\titanic-train.data",
                             delimiter=',', index_col=None, engine='python')

    train_label = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\HW3-CS373\\titanic-train.label",
                              delimiter=',', index_col=None, engine='python')

    test_data = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\HW3-CS373\\titanic-test.data",
                            delimiter=',', index_col=None, engine='python')

    test_label = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\HW3-CS373\\titanic-test.label",
                             delimiter=',', index_col=None, engine='python')

    for att in train_data:
        train_data[att] = train_data[att].fillna(train_data[att].mode()[0])

    p = perceptron(train_data, train_label)
    test_label['survived'] = test_label['survived'].replace({0: -1})

    list1 = []
    for j in range(len(test_data)):
        list1.append(predict(test_data.iloc[j], p[0], p[1]))

    list2 = []
    for j in range(len(test_data)):
        list2.append(hinge_loss(test_data.iloc[j], p[0], p[1], test_label.iloc[j]))

    print("Hinge LOSS = " + str(sum(list2) / len(test_data)))

    print("Test Accuracy = " + str(accuracy(list1, test_label['survived'])))
