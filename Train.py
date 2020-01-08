#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from Models_2D.Model2d import Model2d
from Models_2D.Model_parametes import Model_paras
import pandas as pd
import numpy as np


def train():
    # Read  data set (Train data from CSV file)
    csvimagedata = pd.read_csv('./TrainData.csv')
    traindata = csvimagedata.iloc[:, :].values

    testimagedata = pd.read_csv('./TestData.csv')
    testdata = testimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    for i in range(5):
        perm = np.arange(len(csvimagedata))
        np.random.shuffle(perm)
        traindata = traindata[perm]
    # dataset_name: carotid / cerebral / All
    # net_name: Unet/Unet_bias/Unet_bias_norm/Unet_bias_norm_drop/Unet_res_bias_norm_drop/DASP_Unet
    paras = Model_paras(dataset_name='cerebral', net_name='DASP_Unet')
    Mymodel2D = Model2d(paras)
    Mymodel2D.train(traindata, testdata, Continue_Train=True)

if __name__ == '__main__':
    train()