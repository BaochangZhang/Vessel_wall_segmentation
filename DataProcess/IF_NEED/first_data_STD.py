import SimpleITK as sitk
import os
from DataProcess.utils import get_all_files
import scipy.misc as smc
import numpy as np
import matplotlib.pyplot as plt
"""
IF THE COLLECTED DATA NEED TO STANDARLIZING THE FORMAL
RUN THIS FILE
"""
origin_URL = '/home/zhangbc/Mydataspace/Vesselboudary/carotid/origin_data/4'
OLD_origin_URL = '/home/zhangbc/Mydataspace/Vesselboudary/carotid/OLD_origin_data/4'

def convert():
    if not os.path.exists(origin_URL):
        os.makedirs(origin_URL)
    path_list = get_all_files(OLD_origin_URL)
    for index, filepath in enumerate(path_list):
        rootpath = str(filepath).rsplit('/', 1)[0]
        filename = str(filepath).rsplit('/', 1)[1]
        lablename = str(filename).split('.')[0]+'_Mask.nii'
        labelpath = os.path.join(rootpath, lablename)
        src = sitk.ReadImage(filepath, sitk.sitkInt16)
        srcimg = sitk.GetArrayFromImage(src)
        srcimg = np.squeeze(srcimg)
        srcl = sitk.ReadImage(labelpath, sitk.sitkInt16)
        srclab = sitk.GetArrayFromImage(srcl)
        Label = np.zeros_like(srclab)
        Label[srclab == 1] = 2
        Label[srclab == 2] = 1
        Label = np.squeeze(Label)
        newfilepath = os.path.join(origin_URL, str(filename).split('.')[0]+'.png')
        newlablepath = os.path.join(origin_URL, str(filename).split('.')[0]+'_mask.png')
        smc.imsave(newfilepath, srcimg)
        smc.imsave(newlablepath, Label)


def showdata():
    mask = smc.imread('/home/zhangbc/Mydataspace/Vesselboudary/cerebral/1/0019_mask.png')
    img = smc.imread('/home/zhangbc/Mydataspace/Vesselboudary/cerebral/1/0019.png')
    print(img.shape)
    plt.figure("Image-Mask")
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(mask)
    plt.pause(0.1)
    plt.show()


if __name__ == '__main__':
    showdata()
