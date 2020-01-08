from DataProcess.DataReader import Medical_Image_DataReader as MDReader
import DataProcess.Paras_arg as myflag
import numpy as np
import scipy.io as sio
import csv
import os
from DataProcess.utils import (get_all_files)


def generate_Traindata():

    if not os.path.exists(myflag.Train_URL):
        os.makedirs(myflag.Train_URL)

    if not os.path.exists(myflag.Test_URL):
        os.makedirs(myflag.Test_URL)

    path_list = get_all_files(myflag.origin_URL)
    # shuffle data
    np.random.shuffle(path_list)
    # split data into traindataset and testdataset
    test_ratio = 0.2
    test_num = int(len(path_list) * test_ratio)
    test_data = path_list[:test_num]
    train_data = path_list[test_num:]

    for index, filepath in enumerate(test_data):
        rootpath = str(filepath).rsplit('/', 1)[0]
        filename = str(filepath).rsplit('/', 1)[1]
        patch_id = 0
        data = MDReader(rootpath, filename)
        origin_image = data.image
        origin_label = data.label
        samplename = str(index) + ".mat"
        samplepath = os.path.join(myflag.Test_URL, samplename)
        sio.savemat(samplepath, {'image': origin_image, 'Label': origin_label})
        patch_id += 1

    for index, filepath in enumerate(train_data):
        rootpath = str(filepath).rsplit('/', 1)[0]
        filename = str(filepath).rsplit('/', 1)[1]
        patch_id = 0
        data = MDReader(rootpath, filename)
        origin_image = data.image
        origin_label = data.label
        samplename = str(index) + "_" + str(patch_id) + ".mat"
        samplepath = os.path.join(myflag.Train_URL, samplename)
        sio.savemat(samplepath, {'image': origin_image, 'Label': origin_label})
        patch_id += 1
        flip_mode = [-1, 0, 1]
        for mode_id, mode in enumerate(flip_mode):
            flip_image, flip_label = data.flip_data(mode=mode)
            samplename = str(index) + "_" + str(patch_id) + ".mat"
            samplepath = os.path.join(myflag.Train_URL, samplename)
            sio.savemat(samplepath, {'image': flip_image, 'Label': flip_label})
            patch_id += 1
        random_angles = [90, 180, 270]
        for angle_id, angle in enumerate(random_angles):
            rotated_image, rotated_label = data.rotate_data(angle=angle)
            samplename = str(index) + "_" + str(patch_id) + ".mat"
            samplepath = os.path.join(myflag.Train_URL, samplename)
            sio.savemat(samplepath, {'image': rotated_image, 'Label': rotated_label})
            patch_id += 1
        for i in range(myflag.Num_elastic_trans):
            elastic_image, elastic_label = data.elastic_transform()
            samplename = str(index) + "_" + str(patch_id) + ".mat"
            samplepath = os.path.join(myflag.Train_URL, samplename)
            sio.savemat(samplepath, {'image': elastic_image, 'Label': elastic_label})
            patch_id += 1


def record_data(record_path, data_path1=None, add1=False, data_path2=None, add2=False):
    if os.path.exists(record_path):
        if os.path.isfile(record_path):
            os.remove(record_path)
            with open(record_path, "a+") as record_file:
                writer = csv.writer(record_file)
                writer.writerow(['the-list-of-images-file-path'])
        else:
            print('the given url-info value is wrong')
    else:
        with open(record_path, "a+") as record_file:
            writer = csv.writer(record_file)
            writer.writerow(['the-list-of-images-file-path'])
    if add1:
        for img_file in os.scandir(data_path1):
            if img_file.name.endswith('.mat') and img_file.is_file():
                with open(record_path, "a+") as record_file:
                    writer = csv.writer(record_file)
                    writer.writerow([img_file.path])
    if add2:
        for img_file in os.scandir(data_path2):
            if img_file.name.endswith('.mat') and img_file.is_file():
                with open(record_path, "a+") as record_file:
                    writer = csv.writer(record_file)
                    writer.writerow([img_file.path])


if __name__ == '__main__':
    # generate_Traindata()
    record_data(myflag.Test_img_recodes, add1=False, add2=True, data_path1=myflag.carotid_Test_URL, data_path2=myflag.cerebral_Test_URL)
    record_data(myflag.Train_img_recodes, add1=False, add2=True, data_path1=myflag.carotid_Train_URL, data_path2=myflag.cerebral_Train_URL)