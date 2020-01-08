#!/usr/bin/python
# -*- coding: utf-8 -*-
from Models_2D.Model2d import Model2d
from Models_2D.Model_parametes import Model_paras
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
import sys
import scipy.misc as smc
import os

def eval_one(num):
    num = int(num)
    paras = Model_paras()
    Mymodel2D = Model2d(paras)
    # carotid / cerebral
    testURL = "/home/zhangbc/Mydataspace/Vesselboudary/cerebral/Test_data/"+str(num)+".mat"
    filedata = sio.loadmat(testURL)
    Image = np.asarray(filedata['image'], dtype=np.float32)
    mask = np.asarray(filedata['Label'], dtype=np.float32)
    result, train_accuracy1, train_accuracy2 = Mymodel2D.prediction(Image, mask)
    result = result*127+1
    mask = mask*127+1
    result = np.asarray(result, dtype=np.uint8)
    mask = np.asarray(mask, dtype=np.uint8)
    is_save = True
    if is_save:
        # Cerebral_Result / Carotid_Result
        samplepath = '/home/zhangbc/Mydataspace/Vesselboudary/Cerebral_Result/Unet_basic'
        if not os.path.exists(samplepath):
            os.makedirs(samplepath)
        newfilepath = os.path.join(samplepath, str(num) + '.png')
        newlablepath = os.path.join(samplepath, str(num) + '_Pre.png')
        newGTpath = os.path.join(samplepath, str(num) + '_GT.png')
        smc.imsave(newfilepath, Image)
        smc.imsave(newlablepath, result)
        smc.imsave(newGTpath, mask)
        logfile_path = os.path.join(samplepath, 'log.txt')
        file = open(logfile_path, 'a')
        file.write(str(num) + '.png'+':dice_wall:%.10f, dice_lumen:%.10f' % (train_accuracy1, train_accuracy2))
        file.write('\n')
        file.close()


'''
    roi1 = np.equal(mask, 1).astype('uint8')
    contours1 = measure.find_contours(roi1, 0.5)
    roi2 = np.equal(result, 1).astype('uint8')
    contours2 = measure.find_contours(roi2, 0.5)

    fig, axs = plt.subplots(2, 3, figsize=(30, 18))
    plt.text(-190, -170, 'acc_wall:%.10f, acc_lumen:%.10f' % (train_accuracy1, train_accuracy2), size=15, alpha=1)
    axs[0, 0].imshow(Image, cmap='gray')
    for n, contour in enumerate(contours1):
        axs[0, 0].plot(contour[:, 1], contour[:, 0], linewidth=2)
    axs[0, 0].set_title('GT')
    axs[0, 1].imshow(Image, cmap='gray')
    for n, contour in enumerate(contours2):
        axs[0, 1].plot(contour[:, 1], contour[:, 0], linewidth=2)
    axs[0, 1].set_title('Seg_Result')
    axs[0, 2].imshow(Image, cmap='gray')
    for n, contour in enumerate(contours1):
        axs[0, 2].plot(contour[:, 1], contour[:, 0], linewidth=2, label='GT')
    for n, contour in enumerate(contours2):
        axs[0, 2].plot(contour[:, 1], contour[:, 0], linewidth=2, label='Seg_Result')
    axs[0, 2].legend()
    axs[0, 2].set_title('all method')

    axs[1, 0].imshow(mask)
    axs[1, 0].set_title('GT')
    axs[1, 1].imshow(result)
    axs[1, 1].set_title('Seg_Result')
    axs[1, 2].imshow(Image, cmap='gray')
    axs[1, 2].set_title('Image')
    plt.savefig('/home/zhangbc/wan/Figure_' + str(num) + '.png', dpi=300)
    # plt.show()
'''


def eval_testdataset():
    # dataset_name: carotid / cerebral / All
    # net_name: Unet/Unet_bias/Unet_bias_norm/Unet_bias_norm_drop/Unet_res_bias_norm_drop/DASP_Unet
    paras = Model_paras(dataset_name='cerebral', net_name='DASP_Unet')
    Mymodel2D = Model2d(paras)
    testimagedata = pd.read_csv('./TestData.csv')
    testdata = testimagedata.iloc[:, :].values
    Mymodel2D.eval_testset(testdata)


if __name__ == '__main__':
    # eval_one(sys.argv[1])
    eval_testdataset()
    # eval_one(num=1)
