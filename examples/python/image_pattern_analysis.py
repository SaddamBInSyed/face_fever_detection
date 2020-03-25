import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt

def calc_image_pattern(dirname):

    image_path_list = os.listdir(dirname)

    image_list = []

    for filename in image_path_list:
        if filename.endswith(".tiff"):
            img_path = os.path.join(dirname, filename)

            img = cv2.imread(img_path, -1)

            img = img[:, :, 0]  # take only first channel

            image_list.append(img)

    img_mean = np.stack(image_list, -1).mean(axis=-1)

    img_pattern_mean_subtracted = img_mean - np.mean(img_mean)

    return img_mean, img_pattern_mean_subtracted


if __name__ == '__main__':

    dirname = r'/home/nvidia/Documents/fever_det/2020_25_03__17_35_50'

    img_mean, img_pattern_mean_subtracted = calc_image_pattern(dirname)

    cv2.imwrite('/home/nvidia/Documents/fever_det/2020_25_03__17_35_50/image_pattern.tiff', img_mean)
    cv2.imwrite('/home/nvidia/Documents/fever_det/2020_25_03__17_35_50/image_pattern_mean_subtracted.tiff', image_pattern_mean_subtracted)

    cv2.imshow('image', img_pattern_mean_subtracted)
    cv2.waitKey(0)

    print('Done!')
