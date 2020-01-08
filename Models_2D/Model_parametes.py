#!/usr/bin/python
# -*- coding: utf-8 -*-
from datetime import datetime


class Model_paras(object,):
    def __init__(self, dataset_name, net_name):
        self.H = 128
        self.W = 128
        self.C = 1
        self.n_class = 3
        # dice_loss1/2/3, weighted_dice_loss1/2/3_1/2, cross_entroy, weighted_cross_entroy1/2
        self.cost_name = 'dice_loss3'
        # jaccard is not better than sorensen
        self.loss_type = 'sorensen'
        self.init_learning_rate = 1e-05
        # self.init_learning_rate = 1e-05
        self.decay_rate = 0.90
        self.decay_step = 1000
        self.dropout_rate = 0.8
        self.batch_size = 48
        self.train_epochs = 100
        self.regularizer_rate = 1e-05
        self.norm_name = 'batch'
        self.dataset_name = dataset_name
        self.net_name = net_name
        # carotid / cerebral / All
        self.model_path = "./Mycheckpoint/" + dataset_name + "/" + net_name + "/{}".format(
            datetime.now().strftime("%Y%m%d-%H%M"))
        self.logs_path = self.model_path+"/log"
        self.latest_model = "./latest_model"


if __name__ == '__main__':
    paras = Model_paras()