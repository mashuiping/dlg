# coding: utf-8
# get data

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15,15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28,28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=False)

num_inputs = 784
num_outputs = 10

W = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)

params = [W, b]

for param in params:
    param.attach_grad()

from mxnet import nd
def softmax(X):
    # 不能用batch_size 因为最后可能出现不是batch_size的第0维
    # X = X - X.max(axis=1).reshape((X.shape[0], -1))
    X = X - X.max(axis=1, keepdims=True)
    exp = nd.exp(X)
    for i, l in enumerate(exp):
        for j, e in enumerate(l):
            if math.isnan(e.asscalar()):
                import pdb
                pdb.set_trace()
                print e
    partition = exp.sum(axis=1, keepdims=True)
    # / 对应位置
    return exp / partition

import math
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net):
    acc = 0
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr*param.grad

import sys
sys.path.append("..")
from mxnet import autograd

lr = 1.0
epoch = 10

for e in range(epoch):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()
        SGD(params, lr/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f"
          % (e, train_loss/len(train_data), train_acc/len(train_data), test_acc))

data, label = mnist_test[0:9]
show_images(data)
print ('true labels')
print (get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print ('predicted labels')
print (get_text_labels(predicted_labels.asnumpy()))