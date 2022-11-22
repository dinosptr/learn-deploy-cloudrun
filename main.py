# https://youtu.be/pI0wQbJwIIs
"""

Werkzeug provides a bunch of utilities for developing WSGI-compliant applications. 
These utilities do things like parsing headers, sending and receiving cookies, 
providing access to form data, generating redirects, generating error pages when 
there's an exception, even providing an interactive debugger that runs in the browser. 
Flask then builds upon this foundation to provide a complete web framework.
"""

from flask import Flask, render_template, request, redirect, flash
from PIL import Image
# import os
import tensorflow as tf
import io
import json
import keras
import numpy as np
from datetime import datetime
# import pytz
# import torch
# from matplotlib import pyplot as plt
import numpy as np
# import cv2
# import itertools
import os 
app = Flask(__name__)

model = keras.models.load_model("HAM10000_100epochs.h5")
classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']

@app.route('/')
def index():
    return "asw"

@app.route("/predict", methods=["POST"])
def predict_api():
    if 'file' not in request.files:
        value = {
            "msg": 'Tidak ada file',
        }
        return json.dumps(value)
    file = request.files['file']
    if file.filename == '':
        value = {
            "msg": 'Tidak ada file dipilih untuk diupload'
        }
        return json.dumps(value)
    
    # with open(file,'rb') as f:
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = tf.image.resize(img, [32, 32])

    predictions = model(img)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)

    # timeasia = pytz.timezone('Asia/Jakarta') 
    # now = datetime.now(timeasia)

    return {
        "name": classes[label0],
        # "date": now.strftime("%d/%m/%Y %H:%M:%S")
    }


if __name__ == "__main__":
    app.run(debug=True)
