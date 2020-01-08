import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd
import DataProcess.Paras_arg as myflag
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt


csvdata = pd.read_csv('../'+myflag.Train_img_recodes)
data = csvdata.iloc[:, :].values
filepath = data[3213][0]
filedata = sio.loadmat(filepath)
Mask = np.asarray(filedata['Label'], dtype=np.float32)
Image = np.asarray(filedata['image'], dtype=np.float32)
dis_map_output = np.zeros_like(Mask)
dis_map_output = dis_map_output + 10 * cv2.distanceTransform((Mask == 1).astype(np.uint8),
                                                                            cv2.DIST_L2, 3)
dis_map_output = dis_map_output + cv2.distanceTransform((Mask == 2).astype(np.uint8),
                                                                            cv2.DIST_L2, 3)
fig, axs = plt.subplots(1, 3, figsize=(8, 8))
axs[0].imshow(Image, cmap='gray')
axs[0].set_title('Image')
axs[2].imshow(Mask)
axs[2].set_title('GT')
axs[1].imshow(dis_map_output)
axs[1].set_title('Image')
plt.show()



# label = np.reshape(Mask, (128, 128, 1))
# label = np.tile(label, (1, 1, 3))
# for i in range(3):
#     label[:, :, i] = np.equal(label[:, :, i], i).astype(np.float32)
#
# Y_gt = tf.placeholder("float", shape=[1, 128, 128, 3], name="Output_GT")
# Y_pre = tf.placeholder("float", shape=[1, 128, 128, 3], name="Pre_GT")


# img1 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
# img2 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
# img = tf.concat(values=[img1,img2],axis=3)
# with tf.Session() as sess:
#     img_numpy = sess.run(img)
#     print("out1=", type(img_numpy))
# img_tensor = tf.convert_to_tensor(img_numpy)
# print("out2=", type(img_tensor))
#
# with tf.Session() as sess:
#     img1_numpy = sess.run(img1)
#     print("out1=", type(img1_numpy))
# img1_tensor = tf.convert_to_tensor(img1_numpy)
# print("out2=", type(img1_tensor))

# if cost_name == "distance_loss":
#     Output = tf.argmax(self.Y_pred[:, :, :, :], axis=-1)
#     target = tf.argmax(self.Y_gt[:, :, :, :], axis=-1)
#     with tf.Session() as sess_loss:
#         Output_numpy = sess_loss.run(Output)
#         target_numpy = sess_loss.run(target)
#         Output_dismap = np.zeros_like(Output_numpy)
#         target_dismap = np.zeros_like(target_numpy)
#         for patch_id in range(target_numpy.shape[0]):
#             Output_dismap[patch_id, :, :] = Output_dismap[patch_id, :, :] + \
#                                             self.dynamic_W[0] * cv2.distanceTransform(
#                                                 (Output_numpy[patch_id, :, :] == 1).
#                                                 astype(np.uint8), cv2.DIST_L2, 3)
#             Output_dismap[patch_id, :, :] = Output_dismap[patch_id, :, :] + \
#                                             self.dynamic_W[1] * cv2.distanceTransform(
#                                                 (Output_numpy[patch_id, :, :] == 2).
#                                                 astype(np.uint8), cv2.DIST_L2, 3)
#             target_dismap[patch_id, :, :] = target_dismap[patch_id, :, :] + \
#                                             self.dynamic_W[0] * cv2.distanceTransform(
#                                                 (target_numpy[patch_id, :, :] == 1).
#                                                 astype(np.uint8), cv2.DIST_L2, 3)
#             target_dismap[patch_id, :, :] = target_dismap[patch_id, :, :] + \
#                                             self.dynamic_W[1] * cv2.distanceTransform(
#                                                 (target_numpy[patch_id, :, :] == 2).
#                                                 astype(np.uint8), cv2.DIST_L2, 3)
#         dis_loss_map = np.sum((Output_dismap - target_dismap) * (Output_dismap - target_dismap), axis=(1, 2))
#         dis_loss = np.mean(dis_loss_map) * 0.05
#         print(dis_loss)
#     dis_loss_tensor = tf.convert_to_tensor(dis_loss)
#     return dis_loss_tensor
# import torch
# import numpy as np
# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
#     '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
#     '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
# )
