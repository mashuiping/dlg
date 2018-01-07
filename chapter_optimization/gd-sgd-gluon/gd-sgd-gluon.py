# coding: utf-8
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import numpy as np
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



