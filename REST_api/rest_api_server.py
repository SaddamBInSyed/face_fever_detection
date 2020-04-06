import flask
import random
import base64
import numpy as np
import time
import collections

app = flask.Flask(__name__)
app.config["DEBUG"] = True

homepage = "<h1>REST-API Example</h1><p>This site is a prototype REST-API</p>"
timings = collections.deque([],maxlen=5)

@app.route('/', methods=['GET'])
def home():
    page = homepage
    return page

@app.route("/predict", methods=["POST"])
def predict():
    response = {'status': 'failure', 'msg': 'unknown'}
    global timings
    timings.append(time.time())
    n_calls = len(timings)
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
        except Exception as e:
            msg = "Failed to convert a field: {}".format(e)
            app.logger.error(msg)
            response = {'status': 'failure', 'msg': msg}
            return flask.jsonify(response)

        #TODO: Do some actual work with the data here
        num_predictions = random.randint(0, 10)
        response["id"] = id
        response["faces"] = []
        response["num_predictions"] = num_predictions
        for i in range(num_predictions):
            pred = {}
            pred['top'] = random.randrange(0, height - 1)
            pred['left'] = random.randrange(0, width - 1)
            pred['bottom'] = random.randrange(pred['top'], height)
            pred['right'] = random.randrange(pred['left'], width)
            pred_height = pred['bottom'] - pred['top']
            pred_width = pred['right'] - pred['left']
            pred_ceter_row = pred['top'] + int(pred_height / 2)
            pred_ceter_col = pred['left'] + int(pred_width / 2)
            pred['temp'] = float(temperatures_image[pred_ceter_row, pred_ceter_col])/100.0
            response["faces"].append(pred)

        # indicate that the request was a success
        response['status'] = 'success'
        response['msg'] = ""
    # return the data dictionary as a JSON response
    return flask.jsonify(response)



app.run(host='0.0.0.0', port=5000)
