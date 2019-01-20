import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('resources/data_1d.csv', 'r'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y)) - Y.mean() * X.sum() / denominator
b = (Y.mean() * X.dot(X)) - X.mean() * X.dot(Y) / denominator

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
