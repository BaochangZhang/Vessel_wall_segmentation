import os
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import scipy.misc as smc


class Medical_Image_DataReader():

    def __init__(self, rootpath, filename):  # 构造函数
        self.rootpath = rootpath
        self.filename = filename
        self.maskname = str(filename).split('.')[0]+'_mask.png'
        self.image = self.load_image()
        self.label = self.load_label()

    def load_image(self):
        filepath = os.path.join(self.rootpath, self.filename)
        if not os.path.isfile(filepath):
            print(filepath + 'does not exist!')
            return None
        srcimg = smc.imread(filepath).astype(np.float32)
        return np.squeeze(srcimg)

    def load_label(self):
        filepath = os.path.join(self.rootpath, self.maskname)
        if not os.path.isfile(filepath):
            print(filepath + 'does not exist!')
            return None
        srcimg = smc.imread(filepath).astype(np.float32)
        srcimg[srcimg > 200] = 2
        srcimg[srcimg > 100] = 1
        return np.squeeze(srcimg)

    def elastic_transform(self, alpha=None, sigma=None, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState(None)

        origin_mergedata = np.concatenate((self.image[:, :, np.newaxis], self.label[:, :, np.newaxis]), axis=2)
        shape = origin_mergedata.shape

        if alpha is None:
            alpha = shape[1] * 1
        if sigma is None:
            sigma = shape[1] * 0.07

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        im_merge_t = map_coordinates(origin_mergedata, indices, order=1, mode='reflect').reshape(shape)

        elastic_image = np.squeeze(im_merge_t[:, :, 0:1])
        elastic_label = np.squeeze(im_merge_t[:, :, 1:2])
        elastic_label[elastic_label >= 1.5] = -2
        elastic_label[elastic_label >= 0.5] = -1
        elastic_label[elastic_label >= 0] = 0
        elastic_label = -elastic_label

        return elastic_image, elastic_label

    def flip_data(self, mode):
        if mode not in [-1, 0, 1]:
            mode = 1
        flipped_image = cv2.flip(self.image, mode)
        flipped_label = cv2.flip(self.label, mode)
        flipped_label[flipped_label >= 1.5] = -2
        flipped_label[flipped_label >= 0.5] = -1
        flipped_label[flipped_label >= 0] = 0
        flipped_label = -flipped_label

        return flipped_image, flipped_label

    def rotate_data(self, angle, center=None):
        (h, w) = self.image.shape[:2]
        if center is None:  # 3
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(self.image, M, (w, h))
        rotated_label = cv2.warpAffine(self.label, M, (w, h))
        rotated_label[rotated_label >= 1.5] = -2
        rotated_label[rotated_label >= 0.5] = -1
        rotated_label[rotated_label >= 0] = 0
        rotated_label = -rotated_label
        return rotated_image, rotated_label
