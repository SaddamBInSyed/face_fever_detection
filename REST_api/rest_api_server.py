import gevent
from gevent.monkey import patch_all;
patch_all()
import flask
from gevent.pywsgi import WSGIServer
import base64
import numpy as np
import time
import collections
import sys
sys.path.insert(0, 'RetinaFace')
from RetinaFace import retinaface
import cv2
import logging
sys.path.insert(0, '..')
import examples.python.transformations as transformations
#from examples.python.temperature_histogram import TemperatureHistogram as TempHist



homepage = "<h1>REST-API Example</h1><p>This site is a prototype REST-API</p>"
timings = collections.deque([],maxlen=5)

FACE_DETECTION_THRESH=0.2
FACE_TEMP_PRECENTILE = [0.9, 1.0]

l_ratio = 0.25
r_ratio = 0.75
t_ratio = 0.1
b_ratio = 0.3

IMAGE_SIZE_WH = (288,384)
# def init_detector():
trt_engine_path = '/home/ffd/face_fever_detection/examples/RetinaFace/models/R50_SOFTMAX_VERT.engine'
retina_prefix = ''
# app.logger.info("Loadind detector ...")
detector = retinaface.RetinaFace(prefix=retina_prefix, TRT_engine_path=trt_engine_path, ctx_id=0, network='net3', use_TRT=True, image_size=IMAGE_SIZE_WH)
# app.logger.info("Finish loadind detector with status {}".format("OK" if detector.TRT_init_ok else "Failure"))
# app.logger.info("Warmup detector...")
empty_img = np.zeros([288,384,3]).astype('uint8')
_,_ = detector.detect(empty_img, FACE_DETECTION_THRESH, scales=[1.0], do_flip=False)
# app.logger.info("Detector ready!")
# app.logger.info("Initialing Histogram ...")
hist_calc_interval = 30 * 60  # [sec]
hist_percentile = 0.85
N_samples_for_temp_th = 50
temp_th_nominal = 34.0
buffer_max_len = 3000  #

# temp_hist = TemperatureHistogram(hist_calc_interval=hist_calc_interval,
#                                  hist_percentile=hist_percentile,
#                                  N_samples_for_temp_th=N_samples_for_temp_th,
#                                  temp_th_nominal=temp_th_nominal,
#                                  buffer_max_len=buffer_max_len)
# app.logger.info("Histogram initialized")

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


def calac_temp_median_of_rect(img, box):

    # for temparture calculation - use raw image
    top = int(box[1])
    left = int(box[0])
    bottom = int(box[3])
    right = int(box[2])

    w = box[2] - box[0]
    h = box[3] - box[1]
    hor_start = int(left + l_ratio * w)
    hor_end = int(left + r_ratio * w)
    ver_start = int(top + t_ratio * h)
    ver_end = int(top + b_ratio * h)

    temp_raw = np.median(img[ver_start:ver_end, hor_start:hor_end])
    forehead_bb = [hor_start, ver_start, hor_end, ver_end]
    return temp_raw, forehead_bb

def calac_temp_precentile(img, box):

    top = int(box[1])
    left = int(box[0])
    bottom = int(box[3])
    right = int(box[2])

    hor_start = int(left)
    hor_end = int(right)
    ver_start = int(top)
    ver_end = int(bottom)

    temp_raw = calc_precentile(img[ver_start:ver_end, hor_start:hor_end], q_min=FACE_TEMP_PRECENTILE[0],
                               q_max=FACE_TEMP_PRECENTILE[1])
    return temp_raw


def rotl_coords(point_xy, im_wh):
    # 1. to central coords:
    im_w, im_h = im_wh
    px = point_xy[0] - im_w / 2
    py = point_xy[1] - im_h / 2
    # 2. rotl around (0,0)
    px_r = -py
    py_r = px
    # 2. shift back to - topleft and return:
    return (px_r + im_w / 2, py_r + im_h / 2)


app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    page = homepage
    return page

@app.route("/predict", methods=["POST"])
def predict():
    response = {'status': 'failure', 'msg': 'unknown'}
    # global detector
    global timings
    if not detector.TRT_init_ok:
        msg = "predict requested before init done"
        app.logger.error(msg)
        response = {'status': 'failure', 'msg': msg}
        return flask.jsonify(response)
    timings.append(time.time())
    n_calls = len(timings)-1
    t_total = timings[-1] - timings[0]
    t_rate = n_calls / t_total if t_total != 0 else 0
    app.logger.info("@/predict: last {} calls took {:.2f} seconds. Rate is {:.2f} Hz. (calls/second)".format(n_calls,
                                                                                                     t_total,
                                                                                                     t_rate))
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read fields from the request (detect a form-encoded request):
        id = flask.request.form['id']
        width = flask.request.form['width']
        height = flask.request.form['height']
        temperatures = flask.request.form['temperatures']
        # alternatively: read fields from the request (detect a jason-encoded request):
        # id = flask.request.json['id']
        # width = flask.request.json['width']
        # height = flask.request.json['height']
        # temperatures = flask.request.json['temperatures']

        mandatory_fields = ['id', 'width', 'height', 'temperatures']
        missing_fields = []
        for field in mandatory_fields:
            if eval(field) is None:
                missing_fields.append(field)

        if len(missing_fields) > 0:
            msg = "The following mandatory field(s) was missing from the POST request: {}".format(missing_fields)
            app.logger.error(msg)
            response = {'status': 'failure', 'msg': msg}
            return flask.jsonify(response)
        # casting fields to the correct types:
        try:
            id = int(id)
            width = int(width)
            height = int(height)
            # decode base64 -> bytes array -> ints array -> ints image
            temperatures_byte_array = base64.b64decode(temperatures)
            temperatures_array = np.frombuffer(temperatures_byte_array, dtype='int32')
            temperatures_image = np.reshape(temperatures_array, (height, width))
            temperatures_image_for_human =  temperatures_image.astype(np.float32)/100.
        except Exception as e:
            msg = "Failed to convert a field: {}".format(e)
            app.logger.error(msg)
            response = {'status': 'failure', 'msg': msg}
            return flask.jsonify(response)

        if temperatures_image.shape != detector.image_size_wh:
            msg = "Image and network shape mismatch"
            app.logger.error(msg)
            response = {'status': 'failure', 'msg': msg}
            return flask.jsonify(response)


        # detect and track faces
        image_for_face_detection = temperatures_image_for_human
        faces_list = detector.detect_and_track_faces(image_for_face_detection, FACE_DETECTION_THRESH, scales=[1.0], do_flip=False)

        # update response
        response["id"] = id
        response["faces"] = faces_list

        # im_plot = cv2.merge((temperatures_image, temperatures_image, temperatures_image))
        # # im_raw = rgb.copy()
        # im_plot = im_plot - im_plot.min()
        # im_plot = im_plot.astype(np.float32)
        # im_plot = (im_plot / (im_plot.max()) * 255).astype(np.uint8)


        #temperatures_image = np.rot90(temperatures_image, -1)
        #rgb = cv2.flip(temperatures_image, 1)
        M = transformations.calculate_affine_matrix(rotation_angle=-90, rotation_center=(0, 0), translation=(0, 0), scale=1)
        # M1 = transformations.calculate_affine_matrix(rotation_angle=-90, rotation_center=(0, 0), translation=(0, 0), scale=1)
        # M2 = transformations.calculate_affine_matrix(rotation_angle=0, rotation_center=(0, 0), translation=(0, 0), scale=-1)
        # M = transformations.concatenate_affine_matrices(M2, M1)
        rgb , Mc = transformations.warp_affine_without_crop(temperatures_image.astype(np.float32), M)
        temperatures_image_for_human, _ = transformations.warp_affine_without_crop(temperatures_image_for_human.astype(np.float32), M)
        print(M)

        Mc_inv = transformations.cal_affine_matrix_inverse(Mc)
        rgb = cv2.merge((rgb, rgb, rgb))
        # im_raw = rgb.copy()
        rgb = rgb - rgb.min()
        rgb = rgb.astype(np.float32)
        rgb = (rgb / (rgb.max()) * 255).astype(np.uint8)
        #cv2.imwrite('after_trn_temp.png',rgb)
        app.logger.info('Detection start ...')
        faces, landmarks = detector.detect(rgb, FACE_DETECTION_THRESH, scales=[1.0], do_flip=False)
        app.logger.info('Detection finish ...')

        num_predictions = len(faces) if faces is not None else 0
        print('Num of faces {}'.format(num_predictions))
        response["id"] = id
        response["faces"] = []
        for i in range(num_predictions):
            pred = {}
            box = faces[i]

            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]

            box = np.array([[left, top], [right, bottom]])
            box = transformations.warp_points(box, Mc_inv).flatten()

            scale_fatcor_x= 2.46 # 16.3/6
            scale_fatcor_y = 25.5 / 28.2
            trans_y = - 0

            pred['left'] = scale_fatcor_x * int(box[1])
            pred['top'] = scale_fatcor_y * int(box[0]) + trans_y
            pred['right'] = scale_fatcor_x * int(box[3])
            pred['bottom'] = scale_fatcor_y * int(box[2]) + trans_y

            temp,forehead=calac_temp_median_of_rect(temperatures_image_for_human, faces[i]) 
            temp = float("%.01f" % temp)
            pred['temp'] = temp
            pred['treshold'] = float(36.5)

            response["faces"].append(pred)

            # temp_hist.write_sample(temp=temp, time_stamp=time.time())
            # # calculate temprature histogram
            # if (np.mod(n, temp_hist.hist_calc_interval) == 0):
            #     time_current = time.time()
            #
            #     temp_th = temp_hist.calculate_temperature_threshold()

            # # adding forehead:
            # pred1 = {}
            # box = forehead
            #
            # left = box[0]
            # top = box[1]
            # right = box[2]
            # bottom = box[3]
            #
            # box = np.array([[left, top], [right, bottom]])
            # box = transformations.warp_points(box, Mc_inv).flatten()
            #
            # pred1['left'] = scale_fatcor_x * int(box[1])
            # pred1['top'] = scale_fatcor_y * int(box[0]) + trans_y
            # pred1['right'] = scale_fatcor_x * int(box[3])
            # pred1['bottom'] = scale_fatcor_y * int(box[2]) + trans_y
            # pred1['temp'] = 0
            # pred1['treshold'] = float(36.5)
            # response["faces"].append(pred1)

        # indicate that the request was a success
        response["num_predictions"] = len(response["faces"])
        response['status'] = 'success'
        response['msg'] = ""
    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__=="__main__":
    # init_detector()
    host = '0.0.0.0'
    port = 5000
    # app.run(host='0.0.0.0', port=5000,use_reloader=False)
    WSGIServer((host, port), app).serve_forever()
