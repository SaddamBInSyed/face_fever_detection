import os
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import time

if __name__ == '__main__':

    # set input and output dirs
    input_dir = os.path.join(os.path.dirname(__file__), 'images')
    output_dir = os.path.join(os.path.dirname(__file__), 'images_out_temp_hist')
    os.makedirs(output_dir, exist_ok=True)

    th_face = 0.1  # 0.8
    scales = [1.0]
    flip = False
    display = True

    # initialize detector
    gpuid = -1  # -1 - use cpu
    detector = RetinaFace(model_path='./models/R50', use_TRT=False, epoch=0, ctx_id=gpuid, network='net3')

# iterate over images and detect faces

test_dataset_list = sorted(os.listdir(input_dir))

for n, img_name in enumerate(test_dataset_list):

    # read image
    image_path = os.path.join(input_dir, img_name)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(img.shape)

    # convert to temperature image
    img_temperature = img * 100

    # detect faces
    start_t = time.time()
    # faces, landmarks = detector.detect(img, th_face, scales=scales, do_flip=flip)

    output_list, faces, landmarks, img_out = detector.detect_and_track_faces(img_temperature, th_face, scales=scales, do_flip=flip,
                                                                    rotate90=True, gray2rgb=True, scale_dynamic_range=True, display=display)
    end_t = time.time()
    print(len(faces), end_t-start_t)

    # save image
    if display:
        filename = os.path.join(output_dir, img_name)
        print('writing', filename)
        cv2.imwrite(filename, img_out)
