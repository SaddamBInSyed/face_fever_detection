import os
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import time

thresh = 0.8
scales = [1024, 1980]

count = 1
gpuid = 0
detector = RetinaFace('./models/R50', 0, gpuid, 'net3')
test_dataset = '/home/nvidia/Desktop/thermapp_testing/for pony/'
    # with open(testset_list, 'r') as fr:
    #     test_dataset = fr.read().split()
num_images = len(os.listdir(test_dataset))
test_dataset_list = os.listdir(test_dataset)
for j, img_name in enumerate(test_dataset_list):
  scales = [1024, 1980]

  image_path = test_dataset + img_name
  img = cv2.imread(image_path)
  print(img.shape)
  im_shape = img.shape
  target_size = scales[0]
  max_size = scales[1]
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  #im_scale = 1.0
  #if im_size_min>target_size or im_size_max>max_size:
  im_scale = float(target_size) / float(im_size_min)
  # prevent bigger axis from being more than max_size:
  if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)

  print('im_scale', im_scale)

  scales = [1.0]
  flip = False

  for c in range(count):
    start_t=time.time()
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    end_t = time.time()
    print(c, faces.shape, landmarks.shape, end_t-start_t)

  if faces is not None and len(faces) >0 :
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
      #print('score', faces[i][4])
      box = faces[i].astype(np.int)
      #color = (255,0,0)
      color = (0,0,255)
      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
      if landmarks is not None:
        landmark5 = landmarks[i].astype(np.int)
        #print(landmark.shape)
        for l in range(landmark5.shape[0]):
          color = (0,0,255)
          if l==0 or l==3:
            color = (0,255,0)
          cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    filename = './detector_test_{}.jpg'.format(j)
    print('writing', filename)
    cv2.imwrite(filename, img)

