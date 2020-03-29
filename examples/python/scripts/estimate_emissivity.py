import os
import numpy as np
import cv2
import tiffile
import mouse_crop
from PIL import Image

def calc_mean_image(dirname):

    image_path_list = os.listdir(dirname)

    image_list = []

    for filename in image_path_list:
        if filename.endswith(".tiff"):
            img_path = os.path.join(dirname, filename)

            img = cv2.imread(img_path, -1)
            # img = tiffile.imread(img_path)
            # img = Image.open(img_path)

            img = img[:, :, 0]  # take only first channel

            image_list.append(img)

    mean_img = np.stack(image_list, -1).mean(axis=-1)

    return mean_img

if __name__ == '__main__':

    dirname = r'/home/nvidia/Documents/fever_det/2020_29_03__18_05_03_emissivity_estimation'

    mean_img = calc_mean_image(dirname)

    mean_img_np = mean_img.copy()
    mean_img_np = mean_img_np - mean_img_np.min()
    mean_img_np = (mean_img_np / mean_img_np.max() * 255.).astype(np.uint8)

    # detect black body
    R = mouse_crop.mouse_crop(image_np=mean_img_np, image_raw=mean_img, num_of_crop=2,
                              text='Select black body then suvid')
    coor = R[1]

    # black body
    left, top, right, bottom = coor[0]
    temp_black_body = np.mean(mean_img[top:bottom, left:right])

    # suvid
    left, top, right, bottom = coor[1]
    temp_suvid = np.mean(mean_img[top:bottom, left:right])

    emissivity_suvid = temp_suvid / temp_black_body

    print(emissivity_suvid)

    print('Done!')


