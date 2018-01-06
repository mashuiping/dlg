# coding: utf-8
from mxnet import nd
from mxnet.gluon import nn
from mxnet import init

# 如何对每个层使用不同的初始化函数
# 可以在定义层的时候， 用weight_initializer=init.One() 等来初始化
net1 = nn.Sequential()
with net1.name_scope():
    net1.add(nn.Dense(4, in_units=4, activation="relu", weight_initializer=init.One()))
    net1.add(nn.Dense(4, in_units=4, weight_initializer=init.Xavier()))
print net1

net1.initialize()

# 如果两个层共用一个参数， 那么求梯度的时候会发生什么？
from mxnet import gluon
from mxnet import autograd

net2 = nn.Sequential()
with net2.name_scope():
    net2.add(nn.Dense(4, in_units=4, activation="relu", weight_initializer=init.Xavier()))
    net2.add(nn.Dense(4, in_units=4, activation="relu", params=net2[-1].params))
    net2.add(nn.Dense(1, in_units=4, activation="relu"))

net2.initialize()

trainer = gluon.Trainer(net2.collect_params(), 'sgd', {'learning_rate': 0.1})
a = nd.random_uniform(shape=(4, 4))
with autograd.record():
    y = net2(a)
y.backward()
trainer.step(1)

print net2[0].weight.grad()
print net2[1].weight.grad()