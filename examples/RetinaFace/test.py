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
scales = [1.0]
flip = False

count = 1
gpuid = -1

# initialize detector
detector = RetinaFace(model_path='./models/R50',use_TRT=False, epoch=0,ctx_id=gpuid, network= 'net3')

# set input and output dirs
test_dataset = os.path.join(os.path.dirname(__file__), 'images')
output_dir = os.path.join(os.path.dirname(__file__), 'images_out')
os.makedirs(output_dir, exist_ok=True)

# iterate over images and detect faces
num_images = len(os.listdir(test_dataset))

test_dataset_list = sorted(os.listdir(test_dataset))

for j, img_name in enumerate(test_dataset_list):

  # read image
  image_path = os.path.join(test_dataset, img_name)
  img = cv2.imread(image_path)
  print(img.shape)

  # detect faces
  for c in range(count):
    start_t=time.time()
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    end_t = time.time()
    print(c, faces.shape, landmarks.shape, end_t-start_t)

  # display
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

    filename = os.path.join(output_dir, img_name)
    print('writing', filename)
    cv2.imwrite(filename, img)

