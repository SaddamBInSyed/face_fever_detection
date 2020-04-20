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

    # initialize detector
    gpuid = -1  # -1 - use cpu
    detector = RetinaFace(model_path='./models/R50', use_TRT=False, epoch=0, ctx_id=gpuid, network='net3')

# iterate over images and detect faces

test_dataset_list = sorted(os.listdir(input_dir))

for n, img_name in enumerate(test_dataset_list):

    # read image
    image_path = os.path.join(input_dir, img_name)
    img = cv2.imread(image_path)
    print(img.shape)

    # detect faces
    start_t = time.time()
    # faces, landmarks = detector.detect(img, th_face, scales=scales, do_flip=flip)
    output_list, faces, landmarks = detector.detect_and_track_faces(img, th_face, scales=scales, do_flip=flip,
                                                                    rotate90=False, gray2rgb=False, scale_dynamic_range=True)
    end_t = time.time()
    print(faces.shape, landmarks.shape, end_t-start_t)

    # display
    if faces is not None and len(faces) > 0:

        print('find', faces.shape[0], 'faces')

        # display faces
        for m in range(faces.shape[0]):
            #print('score', faces[i][4])
            box = faces[m].astype(np.int)
            #color = (255,0,0)
            color = (0,0,255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

            # display landmarks
            if landmarks is not None:
                landmark5 = landmarks[m].astype(np.int)
                #print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0,0,255)
                    if l==0 or l==3:
                        color = (0,255,0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    # save image
    filename = os.path.join(output_dir, img_name)
    print('writing', filename)
    cv2.imwrite(filename, img)
