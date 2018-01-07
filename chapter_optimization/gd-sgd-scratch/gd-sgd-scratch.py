# coding: utf-8

# 小批量随机梯度下降。
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

import mxnet as mx
from mxnet import autograd
from mxnet import ndarray as nd
from mxnet import gluon

import random

mx.random.seed(1)
random.seed(1)

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X = nd.random_normal(scale=1, shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(scale=1, shape=y.shape)
dataset = gluon.data.ArrayDataset(X, y)

# 构造迭代器
import random
def data_iter(batch_size):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        j = nd.array(idx[i: min(i + batch_size, num_examples)])
        yield batch_i, X.take(j), y.take(j)

# 初始化模型参数
def init_params():
    w = nd.random_normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params

# 线性回归模型
def net(X, w, b):
    return nd.dot(X, w) + b

# 损失函数
def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt
import numpy as np

def train(batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    w, b = init_params()
    total_loss = [np.mean(square_loss(net(X, w, b), y).asnumpy())]

    for epoch in range(1, epochs+1):
        if epoch > 2:
            lr *= 0.1
        for batch_i, data, label in data_iter(batch_size):
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            sgd([w, b], lr, batch_size)
            if batch_i * batch_size % period == 0:
                total_loss.append(np.mean(square_loss(net(X, w, b), y).asnumpy()))
        print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e"
              % (batch_size, lr, epoch, total_loss[-1]))

    print("w: ", np.reshape(w.asnumpy(), (1, -1)),
          "b: ", b.asnumpy()[0], '\n')

    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('1.png')

# 小批量随机梯度下降
train(batch_size=1, lr=0.2, epochs=3, period=10)
# 梯度下降
train(batch_size=1000, lr=0.999, epochs=3, period=1000)
# 学习率适中
train(batch_size=10, lr=0.2, epochs=3, period=10)
# 学习率太大
train(batch_size=10, lr=5, epochs=3, period=10)
# 学习率太小
train(batch_size=10, lr=0.002, epochs=3, period=10)
