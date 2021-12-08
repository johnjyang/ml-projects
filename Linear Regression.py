import numpy as np
import matplotlib.pyplot as plt

theta = np.random.uniform(-2, 2, (5,))  # Between -2 --> 2, np.array of 5
x = np.array([[2104, 5, 1, 45],
              [1416, 3, 2, 30],
              [1534, 3, 2, 30],
              [852, 2, 1, 36]])
y = np.array([460, 232, 315, 178])


# feature_scale: z --> z
def feature_scale(array, type):
    if type == 'x':
        return (array - np.mean(array, axis=0)) / np.std(array, axis=0)
    else:
        return (array - np.mean(array)) / np.std(array)


x = feature_scale(x, 'x').T
y = feature_scale(y, 'y')

print(x)
print(y)


# f1: x, theta --> y(estimation of y)
def f(x, t):
    list = np.concatenate((np.array([[1, 1, 1, 1]]), x))
    return np.matmul(list.T, t)


print(f(x, theta))


# loss: py, y --> loss
def loss(x, y, t):
    return np.sum((f(x, t) - y) ** 2) / len(x.T)


print(loss(x, y, theta))


# update: x, y, theta --> theta
def update(x, y, t):
    result = np.array([None for i in range(len(t))])
    alpha = 1e-3
    result[0] = t[0] - alpha * 2 * np.sum((f(x, t) - y)) / len(x)
    result[1:] = t[1:] - alpha * 2 * np.sum((f(x, t) - y) * x, axis=1) / len(x)
    return result


print(theta)
print(update(x, y, theta))


# optimize: x y theta --> theta
def optimize(x, y, t):
    old = t
    new = update(x, y, old)

    while loss(x, y, old) > loss(x, y, new):
        old = new
        new = update(x, y, new)

    return new


FinalT = optimize(x, y, theta)

print('o', theta)
print('n', FinalT)

print('\nFinal Loss', loss(x, y, FinalT))


