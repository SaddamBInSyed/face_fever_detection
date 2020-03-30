import signal
import sys
import time
import pyipccap
import os
import datetime
import argparse
import glob

sys.path.insert(0, '../RetinaFace')
from retinaface import RetinaFace
import time
import mouse_crop
import numpy as np

import cv2

from time import sleep

FACE_DETECTION_THRESH = 0.2
# TEMPERATURE_THRESH = 35.0 #38. # FIXME
BODY_TO_SKIN_BIAS = 0. #1.5 #0. #2.2 # FIXME
TEMPERATURE_THRESH = 35.0 + BODY_TO_SKIN_BIAS #38. # FIXME
FACE_EMISSIVITY = 0.98 # 0.92  # 0.98
BLACK_BODY_EMISSIVITY = 1.0
BLACK_BODY_TEMP = [32., 35.] # [36.0, 39.0]
SUVID_EMISSIVITY = 0.9866 * BLACK_BODY_EMISSIVITY #
# SUVID_EMISSIVITY = 1. * BLACK_BODY_EMISSIVITY #
SUVID_TEMP_NOMINAL = 35.0 #37.0
SUVID_BIAS = 0.8 # 0.8
SUVID_TEMP = SUVID_TEMP_NOMINAL + SUVID_BIAS
IM_SHAPE=[384,288,3]
A = 0.01176
B = -2.0

l_ratio = 0.25
r_ratio = 0.75
t_ratio = 0.1
b_ratio = 0.3  # 0.35

temp_measure_type = 'median' #'precentile' # 'median'
FACE_TEMP_PRECENTILE = [0.8, 1.0]

DETECT_CURSUR_TEMP = False

# colormap = None
colormap = 2 # JET

CALIBRATE_FROM_FILE = True #False # False - calibrate form teo black body
ONLINE_CALIBRATION = 'SUVID' # 'BLACK_BODY'
ONETIME_CALIBRATE_DONE = False

scales = [1024, 1980]

name ='/ipc0'
g_calib_black_body = True
g_dumpFiles = False
# g_log2Length = args.l
g_showPic = True
g_dumpPath = '/home/nvidia/Documents/fever_det/'
g_drawGraph = True  # showing the pic implies not drawing the graph

# get image pattern for NUC
subtract_image_pattern = True
dirname = r'/home/nvidia/Documents/fever_det/2020_29_03__18_36_23_NUC_pattern_estimation'

count = 1
gpuid = 0
detector = RetinaFace('../RetinaFace/models/R50', 0, gpuid, 'net3')
_, _ = detector.detect(np.zeros(IM_SHAPE).astype('uint8'), FACE_DETECTION_THRESH, scales=[1.0], do_flip=False)

PY_MAJOR_VERSION = sys.version_info[0]

if PY_MAJOR_VERSION > 2:
    NULL_CHAR = 0
else:
    NULL_CHAR = '\0'


def calib_black_body(im, im_raw, num_of_body=2):
    R = mouse_crop.mouse_crop(image_np=im, image_raw=im_raw, num_of_crop=num_of_body,
                              text='Select 2 black bodies, first with lower temperature')
    GL_mean = []
    for crop in R[0]:
        GL_mean.append(crop.mean())

    A = (GL_mean[0] - GL_mean[1]) / (BLACK_BODY_EMISSIVITY * (BLACK_BODY_TEMP[0] - BLACK_BODY_TEMP[1]))
    B = GL_mean[0] - BLACK_BODY_EMISSIVITY * A * BLACK_BODY_TEMP[0]

    coor = R[1]

    calib_params={'BLACK_BODY_TEMP':BLACK_BODY_TEMP, 'A':A, 'B':B}
    np.save('calibration_params/calib_params.npy',calib_params)

    fname = 'calibration_params/calib_params.txt'
    if not os.path.exists(fname):
        with open(fname, 'a') as f:
            header_str = 'Date-Time, temp1, temp2, emissivity, gray_level1, gray_level2, A, B\n'
            f.write(header_str)

    with open(fname, 'a') as f:
        data_str = '{}, {}, {}, {}, {}, {}, {}, {}\n'.format(datetime.datetime.now().strftime('%Y_%d_%m__%H_%M_%S'), BLACK_BODY_TEMP[0], BLACK_BODY_TEMP[1], BLACK_BODY_EMISSIVITY, GL_mean[0], GL_mean[1], A, B)
        f.write(data_str)

    fname = 'calibration_params/calib_params_{}.npy'.format(datetime.datetime.now().strftime('%Y_%d_%m__%H_%M_%S'))
    np.save(fname, calib_params)

    # np.savetext('calib_params.txt')
    # g_dumpPath = os.path.join(g_dumpPath, datetime.datetime.now().strftime('%Y_%d_%m__%H_%M_%S'))
    return A, B, coor

def calib_offset_online(crop):

    B = GL_mean[0] - BLACK_BODY_EMISSIVITY * A * BLACK_BODY_TEMP[0]

def detect_rectangle(img):
    # Reading same image in another variable and converting to gray scale.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # plt.imshow(img, cmap='gray')

    # Converting image to a binary image (black and white only image).
    _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # plt.imshow(threshold, cmap='gray')

    # Detecting shapes in image by selecting region with same colors or intensity.
    # _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv 3.x
    # import pdb
    # pdb.set_trace()
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv 4.x

    # Searching through every region selected to find the required polygon.
    rect_est = None
    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Shortlisting the regions based on there area.
        if area > 2200 and area < 4200:

            est_polyline = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True),
                                            True)  # [top left, bottom left, bottom right, top right]

            # Checking if the no. of sides of the selected region is 4.
            if (len(est_polyline) == 4):
                #     cv2.drawContours(img2, [approx], 0, (255, 0,), 5)
                rect_est = est_polyline
                break

    if rect_est is not None:
        top = rect_est[0][0, 1]
        left = rect_est[0][0, 0]
        bottom = rect_est[2][0, 1]
        right = rect_est[2][0, 0]
    else:
        top, left, bottom, right = None, None, None, None

    return top, left, bottom, right, rect_est


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

def calc_precentile(x, q_min=0.8, q_max=0.95):

    x_sorterd = np.sort(x.flatten())

    N = len(x_sorterd) - 1

    ind_min = int(np.floor(q_min * N))
    ind_max = int(np.floor(q_max * N))

    if ind_min < ind_max:
        y = np.mean(x_sorterd[ind_min:ind_max])
    else:
        y = x_sorterd[ind_max]

    return y

mouseX, mouseY, temp_mouse = 0, 0, 0.

def get_mouse_loc(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            mouseX,mouseY = x,y
            print('mouse_loc:{},{}'.format(x,y))

        except:
            pass


# cv2.namedWindow('image')
# cv2.setMouseCallback('image',get_mouse_loc)

# def main():
    # Instantiate the parser
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('name', type=str,
    #                     help='part of shared memory name')
    #
    # parser.add_argument('-noshow', action='store_true', default=False,
    #                     help='Specify -nowshow flag if you don\'t want to show the images on the screen')
    #
    # parser.add_argument('-d', action='store_true', default=False,
    #                     help='Specify to dump the incoming frames to files')
    #
    # parser.add_argument('-o', type=str, default="/media/nano/Elements/",
    #                     help='if -d is specified => will save images to the specified directory')
    #
    # # Optional argument
    # parser.add_argument('-l', type=int,
    #                     help='The log 2 of number of ipc buffers to allocate.\nValid values are in the range of 0 for one buffer and 4 for 16 buffers.\nThe default is 2.')

    # args = parser.parse_args()

img_mean, img_pattern_mean_subtracted = calc_image_pattern(dirname)


print("Params:")
print("name:          {}".format(name))
print("Dump files:    {}".format(g_dumpFiles))
if g_dumpFiles:
    g_dumpPath = os.path.join(g_dumpPath, datetime.datetime.now().strftime('%Y_%d_%m__%H_%M_%S'))
    print("Dump path      {}".format(g_dumpPath))
    try:
        os.mkdir(g_dumpPath)
        g_dumpPath_display = os.path.join(g_dumpPath, 'display')
        os.mkdir(g_dumpPath_display)
        g_dumpPath_png = os.path.join(g_dumpPath, 'png_im')
        os.mkdir(g_dumpPath_png)
    except:
        import traceback;
        traceback.print_exc()
        print("ERROR: can not create folder: {}".format(g_dumpPath))
        import pdb;
        pdb.set_trace()
print("Show image:    {}".format(g_showPic))

g_log2Length = 2

if g_log2Length < 0:
    print("Log 2 Length of %d is less than 0. 0 will be used\n", g_log2Length)
    g_log2Length = 2

if g_log2Length > 4:
    print("Log 2 Length of %d is greater than 4. 4 will be used\n", g_log2Length)
    g_log2Length = 4;

print("g_log2Length:  {}".format(g_log2Length))

nOutputImageQ = 1 << g_log2Length
print("nOutputImageQ: {} ".format(nOutputImageQ))

thermapp = pyipccap.thermapp(None, nOutputImageQ)
# thermapp = pyipccap.thermapp(name,nOutputImageQ)

print "Open shared memory..."

thermapp.open_shared_memory()

print "Shared memory opened"

rate_list = []
frame_count = 0
prev = 0
last_msg = time.time()
avg_accum = 0
avg_count = 0
total_misses = 0
try:
    while True:

        data = thermapp.get_data()
        if data is None:
            continue

        # We have new data

        # time calc
        now = time.time()
        diff = now - last_msg
        last_msg = now
        if diff > 0:
            msg_rate = 1.0 / diff
        else:
            msg_rate = 0.0
        miss = thermapp.imageId - prev - 1
        prev = thermapp.imageId

        avg_max = 60  # count avg every avg_max frames
        if avg_count < avg_max:
            avg_accum += msg_rate
            avg_count += 1
            print("Got {}, dim ({},{}), missed = {}, rate = {} Hz".format(thermapp.imageId, thermapp.imageWidth,
                                                                          thermapp.imageHeight, miss,
                                                                          round(msg_rate, 2)))
        else:
            avg_rate = avg_accum / avg_max
            print("Got {}, dim ({},{}), missed = {}, rate = {} Hz, avg_rate = {} Hz".format(thermapp.imageId,
                                                                                            thermapp.imageWidth,
                                                                                            thermapp.imageHeight,
                                                                                            miss,
                                                                                            round(msg_rate, 2),
                                                                                            round(avg_rate, 2)))
            avg_accum = 0
            avg_count = 0

        if (frame_count) > 10:
            rate_list.append(msg_rate)
            total_misses += miss

        if g_showPic:
            font = cv2.FONT_HERSHEY_PLAIN
            img_st = data[64:]
            rgb = np.frombuffer(img_st, dtype=np.uint16).reshape(thermapp.imageHeight, thermapp.imageWidth)
            if subtract_image_pattern:
                rgb = (rgb - img_pattern_mean_subtracted).astype(np.uint16)
            # rgb = np.rot90(rgb,3)
            # rgb = np.uint8(rgb)

            rgb = cv2.merge((rgb, rgb, rgb))
            im_raw = rgb.copy()
            im = rgb - rgb.min()
            im = im.astype('float')
            im = (im / (im.max()) * 255).astype('uint8')

            if not ONETIME_CALIBRATE_DONE: # first time calibration setup

                ONETIME_CALIBRATE_DONE = True

                if CALIBRATE_FROM_FILE:  # calibrate from file
                    calib_params = np.load('calibration_params/calib_params.npy', allow_pickle=True)
                    calib_params = calib_params.flatten()[0]
                    A = calib_params['A']
                    B = calib_params['B']
                else: # calibrate from image taken now
                    A, B, coor = calib_black_body(im=im, im_raw=im_raw, num_of_body=2)
                    coor_black_body = coor


                if ONLINE_CALIBRATION== 'SUVID':
                    R = mouse_crop.mouse_crop(image_np=im, image_raw=im_raw, num_of_crop=1,
                                              text='Select reference object (Suvid)')
                    coor_suvid = R[1]



            scales = [1024, 1980]

            im_plot = cv2.resize(im, (384 * 2, 288 * 2))
            # im_plot = im
            if colormap is not None:
                im_plot = cv2.applyColorMap(im_plot, colormap)  # 2 is type of color map

            print(im.shape)
            im_shape = im.shape
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            # im_scale = 1.0
            # if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            print('im_scale', im_scale)
            scales = [1.0]
            flip = False
            start_t = time.time()
            faces, landmarks = detector.detect(im, FACE_DETECTION_THRESH, scales=scales, do_flip=flip)
            end_t = time.time()
            print(faces.shape, landmarks.shape, end_t - start_t)

            if not CALIBRATE_FROM_FILE: # show temperature of black body

                for c in coor_black_body:

                    color = (255, 0, 0)
                    left, top, right, bottom = c
                    # if temp_measure_type == 'median':
                    #     ref_temp_raw = np.median(im_raw[top:bottom, left:right, 0])
                    # else:
                    #     ref_temp_raw = calc_precentile(im_raw[top:bottom, left:right, 0], q_min=FACE_TEMP_PRECENTILE[0],
                    #                                    q_max=FACE_TEMP_PRECENTILE[1])

                    ref_temp_raw = np.mean(im_raw[top:bottom, left:right, 0])
                    ref_temp = (ref_temp_raw - B) / (BLACK_BODY_EMISSIVITY * A)
                    cv2.rectangle(im_plot, (2 * left, 2 * top), (2 * right, 2 * bottom), color, 3)
                    color = (0, 255, 0)
                    cv2.putText(im_plot, '{}'.format(np.round(ref_temp, 1)), (2 * left, 2 * top - 20), font, 2, color, 3)

            if ONLINE_CALIBRATION: # show temperature of suvid, and estimate B

                for c in coor_suvid:
                    color = (255, 0, 0)
                    left, top, right, bottom = c
                    # if temp_measure_type == 'median':
                    #     ref_temp_raw = np.median(im_raw[top:bottom, left:right, 0])
                    # else:
                    #     ref_temp_raw = calc_precentile(im_raw[top:bottom, left:right, 0], q_min=FACE_TEMP_PRECENTILE[0],
                    #                                    q_max=FACE_TEMP_PRECENTILE[1])

                    ref_temp_raw = np.mean(im_raw[top:bottom, left:right, 0])

                    B = ref_temp_raw - SUVID_EMISSIVITY * A * SUVID_TEMP

                    ref_temp = (ref_temp_raw - B) / (SUVID_EMISSIVITY * A) 

                    cv2.rectangle(im_plot, (2 * left, 2 * top), (2 * right, 2 * bottom), color, 3)
                    color = (0, 255, 0)
                    cv2.putText(im_plot, '{}'.format(np.round(ref_temp, 1)), (2 * left, 2 * top - 20), font, 2, color,
                                3)


                # color = (255, 0, 0)
                # left, top, right, bottom = coor[0]
                # if temp_measure_type == 'median':
                #     ref_temp_raw = np.median(im_raw[top:bottom, left:right, 0])
                # else:
                #     ref_temp_raw = calc_precentile(im_raw[top:bottom, left:right, 0], q_min=FACE_TEMP_PRECENTILE[0], q_max=FACE_TEMP_PRECENTILE[1])
                # ref_temp = (ref_temp_raw - B) / (BLACK_BODY_EMISSIVITY * A)
                # cv2.rectangle(im_plot, (2 * left, 2 * top), (2 * right, 2 * bottom), color, 3)
                # color = (0, 255, 0)
                # cv2.putText(im_plot, '{}'.format(np.round(ref_temp, 1)), (2 * left, 2 * top - 20), font, 2, color,
                #             3)
                #
                # color = (255, 0, 0)
                # left, top, right, bottom = coor[1]
                # if temp_measure_type == 'median':
                #     ref_temp_raw = np.median(im_raw[top:bottom, left:right, 0])
                # else:
                #     ref_temp_raw = calc_precentile(im_raw[top:bottom, left:right, 0], q_min=FACE_TEMP_PRECENTILE[0], q_max=FACE_TEMP_PRECENTILE[1])
                #
                # cv2.rectangle(im_plot, (2 * left, 2 * top), (2 * right, 2 * bottom), color, 3)
                # ref_temp = (ref_temp_raw - B) / (BLACK_BODY_EMISSIVITY * A)
                # color = (0, 255, 0)
                # cv2.putText(im_plot, '{}'.format(np.round(ref_temp, 1)), (2 * left, 2 * top - 20), font, 2, color,
                #             3)

                # ref_temp_raw = np.max(np.sort(im_raw[:, :, 0].reshape([-1]))[-100:])
                # ref_temp_raw = im_raw.max()
                # ref_temp = (ref_temp_raw - 14336) * 0.00652 + 110.7
                print('black body not found :(')
                status_text = 'black body not found'
                # ref_temp = (ref_temp_raw - B) / (BLACK_BODY_EMISSIVITY * A)
                # cv2.putText(im_plot, '{}'.format(ref_temp), (2 * left, 2 * top - 20), font, 2, color, 3)

            print(ref_temp_raw)
            print(ref_temp)

            # cv2.rectangle(im_plot, (2 * left, 2 * top), (2 * right, 2 * bottom), color, 3)
            # )

            # cv2.putText(im_plot, status_text, (10, 10), font, 3,
            #             color, 3)

            if faces is not None and len(faces) > 0:
                print('find', faces.shape[0], 'faces')
                for i in range(faces.shape[0]):
                    # print('score', faces[i][4])
                    box = faces[i].astype(np.int)
                    # color = (255,0,0)
                    color = (0, 0, 255)
                    # temp_raw = im_raw[box[0]:box[2],box[1]:box[3],0].mean()
                    box_draw = 2 * box

                    top_draw = box_draw[1]
                    left_draw = box_draw[0]

                    w_draw = box_draw[2] - box_draw[0]
                    h_draw = box_draw[3] - box_draw[1]

                    hor_start_draw = int(left_draw + l_ratio * w_draw)
                    hor_end_draw = int(left_draw + r_ratio * w_draw)
                    ver_start_draw = int(top_draw + t_ratio * h_draw)
                    ver_end_draw = int(top_draw + b_ratio * h_draw)

                    # for temparture calculation - use raw image
                    top = box[1]
                    left = box[0]
                    bottom = box[3]
                    right = box[2]

                    w = box[2] - box[0]
                    h = box[3] - box[1]

                    if temp_measure_type == 'median':

                        hor_start = int(left + l_ratio * w)
                        hor_end = int(left + r_ratio * w)
                        ver_start = int(top + t_ratio * h)
                        ver_end = int(top + b_ratio * h)

                        temp_raw = np.median(im_raw[ver_start:ver_end, hor_start:hor_end, 0])

                        # display rectangle on forehead
                        cv2.rectangle(im_plot, (hor_start_draw, ver_start_draw), (hor_end_draw, ver_end_draw),
                                      (255, 255, 255), 2)

                    else:

                        hor_start = int(left)
                        hor_end = int(right)
                        ver_start = int(top)
                        ver_end = int(bottom)

                        temp_raw = calc_precentile(im_raw[ver_start:ver_end, hor_start:hor_end, 0], q_min=FACE_TEMP_PRECENTILE[0], q_max=FACE_TEMP_PRECENTILE[1])

                    # a= 1.0/(FACE_EMISSIVITY*A)
                    # b= B*FACE_EMISSIVITY*a
                    # temp_raw = temp_raw/FACE_EMISSIVITY
                    temp = (temp_raw - B) / (FACE_EMISSIVITY * A)

                    temp += BODY_TO_SKIN_BIAS  #

                    # temp += 1 # FIXME!!!
                    # temp =(temp_raw-B)/(BLACK_BODY_EMISSIVITY*A) # FIXME!!!!

                    # temp = (temp_raw - 14336) * 0.00652 +110.7

                    # is_temp_ok = temp_raw <= ref_temp_raw * FACE_EMISSIVITY
                    is_temp_ok = temp <= 38.0

                    text_ = np.round(temp, 1)

                    if is_temp_ok:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(im_plot, (box_draw[0], box_draw[1]), (box_draw[2], box_draw[3]), color, 3)

                    # cv2.putText(im_plot, '{}'.format(text_), (box_draw[0], box_draw[1] - 20), font, 2, color, 3)
                    cv2.putText(im_plot, '{}'.format(text_), (box_draw[0], box_draw[1] - 20), font, 4, color, 3)

                    # cv2.putText(im_plot, "{:.1f}".format(temp), (box_draw[0], box_draw[1] - 20), font, 1,
                    #             color, 3)
                    # if landmarks is not None:
                    #     landmark5 = landmarks[i].astype(np.int)
                    #     # print(landmark.shape)
                    #     for l in range(landmark5.shape[0]):
                    #         color = (0, 0, 255)
                    #         # if l == 0 or l == 3:
                    #         #     color = (0, 255, 0)
                    #         cv2.circle(im, (landmark5[l][0], landmark5[l][1]), 1, color, 1)
                    # cv2.circle(im, (landmark5[2][0], landmark5[2][1]), 1, color, 2)

            # mouse temperature
            if DETECT_CURSUR_TEMP:
                mouse_x = int(0.5 * mouseX)
                mouse_y = int(0.5 * mouseY)
                mouse_radius = 3
                temp_mouse_raw = np.mean(im_raw[(mouse_y-mouse_radius):(mouse_y+mouse_radius), (mouse_x-mouse_radius):(mouse_x+mouse_radius), 0])
                temp_mouse = (temp_mouse_raw - B) / (BLACK_BODY_EMISSIVITY * A)
                temp_mouse_text = 'Cursur Temperatur: {}'.format(np.round(temp_mouse, 1))
                color = (0, 255, 0)
                cv2.putText(im_plot, temp_mouse_text, (10, 50), font, 3, color, 3)


            cv2.imshow('image', im_plot)

            if DETECT_CURSUR_TEMP and (frame_count == 0):
                cv2.setMouseCallback('image', get_mouse_loc)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if g_dumpFiles:
            frame_name = os.path.join(g_dumpPath, "{}.tiff".format(thermapp.imageId))
            print("save image: {}".format(frame_name))
            cv2.imwrite(frame_name, im_raw)

            frame_name = os.path.join(g_dumpPath_png, "{}.png".format(thermapp.imageId))
            print("save image: {}".format(frame_name))
            cv2.imwrite(frame_name, im)

            frame_name = os.path.join(g_dumpPath_display, "{}.png".format(thermapp.imageId))
            cv2.imwrite(frame_name, im_plot)

        frame_count += 1  # captured frames counter

except KeyboardInterrupt:
    print("Caught ya!")

if g_drawGraph and not g_showPic:
    print("Finished. Now drawing the graph")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(rate_list, label='Rate[Hz]')
    avg = sum(rate_list) / len(rate_list)
    ax.axhline(y=avg, color='r', linestyle='-', label='AVG=' + str(round(avg, 2)))
    ax.plot([], color='r', label='Misses=' + str(total_misses))
    ax.legend(loc='uppder left')
    ax.set_xlabel('time')
    ax.set_ylabel('rate[Hz]')
    plt.show()




# if __name__ == '__main__':
#     main()
