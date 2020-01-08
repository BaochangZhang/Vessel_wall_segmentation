import numpy as np
import pandas as pd
import DataProcess.Paras_arg as myflag
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import measure


csvdata = pd.read_csv('../'+myflag.Train_img_recodes)
data = csvdata.iloc[:, :].values
filepath = data[5443][0]
print(filepath)
filedata = sio.loadmat(filepath)
Image = np.asarray(filedata['image'], dtype=np.float32)
Mask = np.asarray(filedata['Label'], dtype=np.float32)

roi1 = np.equal(Mask, 1).astype('uint8')
contours1 = measure.find_contours(roi1, 0.5)

fig, axs = plt.subplots(1, 3, figsize=(8, 8))
axs[0].imshow(Image, cmap='gray')
for n, contour in enumerate(contours1):
    axs[0].plot(contour[:, 1], contour[:, 0], linewidth=2)
axs[0].set_title('Image_GT-contour')
axs[2].imshow(Mask)
axs[2].set_title('GT')
axs[1].imshow(Image, cmap='gray')
axs[1].set_title('Image')
plt.show()
