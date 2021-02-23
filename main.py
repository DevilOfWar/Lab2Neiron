import numpy as np


def transform(x, n):
    if x < n:
        return 0
    else:
        return 1


def f(x):
    return 1/(1 + np.exp(-x))


def df(x):
    return np.exp(-x) / ((1 + np.exp(-x)) ** 2)


def go_foward(input):
    sum = np.dot(W1, input)
    out = np.array([f(x) for x in sum])
    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)


def train(epoch):
    global W2, W1
    lmd = 0.025
    N = 500
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]
        y, out = go_foward(x[0:6])
        e = y - x[-1]
        delta = e*df(y)
        W2[0] = W2[0] - lmd * delta * out[0]
        W2[1] = W2[1] - lmd * delta * out[1]
        delta2 = W2 * delta * df(out)
        W1[0] = W1[0] - lmd * delta2[0] * np.array(x[0:6])
        W1[1] = W1[1] - lmd * delta2[1] * np.array(x[0:6])


epoch = [[0, 0, 0, 0, 0, 0, 0]]
n = [1, 4, 30, 4, 12, 1]
flag = False
with open("tests.txt", "r") as file:
    for line in file:
        list = line.split(" ")
        if flag is False:
            flag = True
        else:
            epoch.append([0, 0, 0, 0, 0, 0, 0])
        index = 0
        for element in list:
            if element != "\n":
                element_float = float(element)
                if index < 6:
                    epoch[-1][index] = transform(element_float, n[index])
                else:
                    epoch[-1][index] = element_float
                index = index + 1
epoch = np.array(epoch)
W1 = np.array([[-0.11, 0.31, 0.47, 0.29, -0.5, 0.47], [0.4, 0.2, 0.26, -0.3, -0.47, -0.1]])
W2 = np.array([0.5, -0.5])
train(epoch)
for x in epoch:
    y, out = go_foward(x[0:6])
    print(f"Выходное значение НС: {y} => {x[-1]}")
