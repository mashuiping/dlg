# coding: utf-8
from find_best_lr.find_best_lr import  GluonFindBestLr
import mxnet as mx
from mxnet.gluon import nn

net = nn.Sequential()
drop_prob1 = 0.2
drop_prob2 = 0.5

with net.name_scope():
    net.add(nn.Flatten())
    # 第一层全连接。
    net.add(nn.Dense(256, activation="relu"))
    # 在第一层全连接后添加丢弃层。
    net.add(nn.Dropout(drop_prob1))
    # 第二层全连接。
    net.add(nn.Dense(256, activation="relu"))
    # 在第二层全连接后添加丢弃层。
    net.add(nn.Dropout(drop_prob2))
    net.add(nn.Dense(10))
net.initialize(ctx=mx.gpu())

import sys
sys.path.append('..')
import utils
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.5})

# must set lr to a low level
trainer.set_learning_rate(1e-5)
find_best_lr = GluonFindBestLr(trainer)

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        data = data.as_in_context(mx.gpu())
        label = label.as_in_context(mx.gpu())
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

        # fine best lr
        find_best_lr.add_lr(trainer.learning_rate)
        find_best_lr.add_loss(nd.mean(loss).asscalar())
        find_best_lr.set_lr()
        find_best_lr.draw_and_exit()


        test_acc = utils.evaluate_accuracy(test_data, net, ctx=mx.gpu())

        print("Epoch %d. Lr: %f, Loss: %f, Best_loss: %f, Train acc %f, Test acc %f" % (
            epoch, trainer.learning_rate, nd.mean(loss).asscalar(), find_best_lr.best_loss,
            train_acc/len(train_data), test_acc))

    # print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
    #     epoch, train_loss/len(train_data),
    #     train_acc/len(train_data), test_acc))