# import the necessary packages
import requests
import random
import base64
import json

SERVER_REST_API_URL = "http://10.53.128.56:5000/predict"
IMAGE_PATH = "temp_image_wh_384_288_1.raw"

# load the input image and construct the payload for the request
image_data = open(IMAGE_PATH, "rb").read()
image_data_b64 = base64.b64encode(image_data)
image_id = random.randint(0, 2**20)
image_width = 384
image_height = 288

payload = {"id":  image_id,
           "width": image_width ,
           "height": image_height,
           "temperatures": image_data_b64 }

# submit the request as application/xxx-form-url-encoded contnt-type
res = requests.post(SERVER_REST_API_URL, data=payload)
# # submit the request as application/json contnt-type
payload["temperatures"] = image_data_b64.decode('utf-8')
# res = requests.post(SERVER_REST_API_URL, json=payload)
# print the response:
print(json.dumps(res.json(), indent=4))
