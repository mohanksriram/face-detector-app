import flask
from flask import Flask, render_template
from io import BytesIO
from io import StringIO
from typing import List, Dict, Union, ByteString, Any
import requests
import base64
from PIL import Image
import mtcnn
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from keras.models import load_model
from flask import send_file
import pickle as p
import numpy as np
import os
import time
import sys
import cv2

sys.path.append("services")
from services import camera_source, face_detector, face_describer, face_verifier

enable_kmeans = True
RTSP_URL = 'rtsp://admin:admin@192.168.1.117:554'
last_detected_face = 'data/last_face.png'


def serve_pil_image(filename):
    img_io = StringIO()
    # pil_img.save(img_io, 'JPEG', quality=70)
    # img_io.seek(0)
    return send_file(filename, mimetype='image/jpeg')
    #return send_file(img_io, mimetype='image/jpeg')


app = Flask(__name__, static_folder="build/static",
 template_folder="build")


# load the describer model
facenet_model = face_describer.FaceDescriber()
face_classifer = face_verifier.FaceVerifier()


@app.route("/")
def root():
    return render_template('index.html')
    
@app.route("/admin/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/logo")
def serve_img():
    filename = f'data/custom/train/nobody/image_0.png'
    img = Image.open(filename)
    print(f'image: {img}')
    return serve_pil_image(filename)

@app.route("/lastFace/<path:path>")
def serve_face(path):
    print(f'serving last detected face!')
    return serve_pil_image(last_detected_face)


@app.route('/serveImage/<path:path>')
def serve_latest(path):
    print(f'serving matching face!')
    filename = f'data/custom/train/{path}/image_0.png'
    img = Image.open(filename)
    return serve_pil_image(filename)


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        overall_time = time.time()
        start_time = time.time()
        
        print('received a get request')
        data_url = flask.request.args.get("url")
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        img = Image.open(BytesIO(body))

        # camera = camera_source.CameraSource(0)
        # img = camera.getFrame()

        # cv2.imwrite(last_detected_face, img)
        end_time = time.time() - start_time
        print('Time for capturing image: {}'.format(end_time))


        start_time = time.time()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(img)

        # last_detected_face = image

        end_time = time.time() - start_time
        print('Time for cv2 to pil image: {}'.format(end_time))


        # Face detection
        detector = face_detector.FaceDetector()
        face, face_rect = detector.getFaces(img)

        print(f'face_rect: {face_rect}')
        
        if len(face) == 0:
            ## No face detected
            compute_time = time.time() - overall_time
            prediction_names = ['nobody']
            res = {'svc_class': prediction_names[0], 'confidence': 100}
            res.update({'predictions': prediction_names, 'probs': [1], 'compute_time': compute_time, 'face_rect': [0, 0, 0, 0]})
            return flask.jsonify(res)


        print('Detected Face!')

        # Generate Embedding'
        embedding = facenet_model.getEmbedding(face)

        start_time = time.time()
        in_encoder = Normalizer(norm='l2')
        embedding = np.expand_dims(embedding, axis=0)
        
        embedding = in_encoder.transform(embedding)

        res = face_classifer.getPredictions(embedding)
        res.update({'face_rect': face_rect})
        
        print(res)
        return flask.jsonify(res)

    res = []
    return flask.jsonify(res)


def before_request():
    app.jinja_env.cache = {}    



if __name__ == '__main__':
    int()
    port = os.environ.get('PORT', 5000)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, host='0.0.0.0', port=port)