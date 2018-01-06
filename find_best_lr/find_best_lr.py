# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

class BasicFindBestLr():
    """
    a epoch scope:
        find_best_lr.add_lr(...)
        find_best_lr.add_loss(nd.mean(loss).asscalar())
        find_best_lr.set_lr()
        find_best_lr.draw_and_exit()
    """
    def __init__(self, trainer, exit_loss_factor=4, exit_lr=1.0, best_loss=1e9):
        self.lrs = []
        self.losses = []
        self.exit_loss_factor = exit_loss_factor
        self.exit_lr = exit_lr
        self.best_loss = best_loss
        self.best_lr = 1e-9
        self.trainer = trainer

    def find_best_lr(self, lr, loss):
        self._add_lr(lr)
        self._add_loss(loss)
        self._set_best_loss_best_lr(loss, lr)
        self._set_lr()
        self._draw_and_exit()


    def _add_lr(self, lr):
        self.lrs.append(lr)

    def _add_loss(self, loss):
        self.losses.append(loss)

    def _set_best_loss_best_lr(self, loss, lr):
        if self.best_loss > loss:
            self.best_loss = loss
            self.best_lr = lr

    def _set_lr(self):
        # 继承需要重写set_lr方法
        pass

    def _draw_and_exit(self):
        if self.losses[len(self.losses)-1] > self.exit_loss_factor * self.best_loss or\
                self.lrs[len(self.lrs)-1] > self.exit_lr:
            plt.figure()
            plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.plot(np.log(self.lrs), self.losses)
            # plt.plot(self.lrs, self.losses)
            plt.savefig('lr_loss_' + str(self.best_lr) + '.png')
            plt.show()
            plt.figure()
            plt.xlabel('num iterations')
            plt.ylabel('learning rate')
            plt.plot(self.lrs)
            plt.savefig('numIter_lr.png')
            exit(1)

class GluonFindBestLr(BasicFindBestLr):
    """
    add a gluon example !!!
    """
    def __init__(self, trainer, factor=1.12):
        BasicFindBestLr.__init__(self, trainer)
        # factor should be large than 1
        # to enable lr increase
        self.factor = factor

    def _set_lr(self):
        self.trainer.set_learning_rate(self.trainer.learning_rate * self.factor)
