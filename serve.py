
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
from scipy.stats import mode
from sklearn.cluster import KMeans
import os
import time
import sys
import cv2

sys.path.append("services")
from services import camera_source, face_detector, face_describer

enable_kmeans = True
RTSP_URL = 'rtsp://admin:admin@192.168.1.117:554'
last_detected_face = 'data/last_face.png'


def serve_pil_image(filename):
    img_io = StringIO()
    # pil_img.save(img_io, 'JPEG', quality=70)
    # img_io.seek(0)
    return send_file(filename, mimetype='image/jpeg')
    #return send_file(img_io, mimetype='image/jpeg')

def sim_predict(model, new_img_f, kmean_classes, top_n=1, n_classes=5):
    # cluster labels do not match with actual order of train data. so find indices to reorder cluster centres
    kmeans = model
    steps = np.linspace(0, len(kmeans.labels_), num=n_classes+1)
    orig_labels = []
    last_val = 0
    for i in steps[1:]:
        cluster_labels = kmeans.labels_[last_val:int(i)]
        last_val = int(i)
        orig_labels += [mode(cluster_labels)[0][0]]

    # new_map = {}
    # for i, label in enumerate(encode_labels):
    #     new_map[]


    relabeled = kmeans.cluster_centers_[orig_labels]
    sims = np.array([])
    for i in range(relabeled.shape[0]):
        sim = np.dot(relabeled[i],new_img_f)
        sims = np.append(sims,sim)
    sims_top_n = sims.argsort()[-top_n:][::-1]
    classes = sims_top_n

    classes = [kmean_classes[val] for val in classes]
    
    #print(f'new_classes: {classes}')
    probs = sims[sims_top_n]
    #print(f'classes: {classes}')
    return classes, probs

app = Flask(__name__, static_folder="build/static",
 template_folder="build")


# load the describer model
facenet_model = face_describer.FaceDescriber()

start_time = time.time()

filename = 'models/kmeans_model.sav'
kmeans_model = pickle.load(open(filename, 'rb'))

filename = 'models/svc_model.sav'
svc_model = pickle.load(open(filename, 'rb'))

        #    classifier_model = pickle.load(open(filename, 'rb'))
kmean_classes = None
with open("models/kmeans_classes.txt", "rb") as fp:   # Unpickling
    kmean_classes = pickle.load(fp)

svc_classes = np.load('models/svc_classes.npy')

end_time = time.time() - start_time
print('Time for loading classifier: {}'.format(end_time))



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

        # Perform prediction
        #if not classifier_model:
        #    filename = 'models/classifier_model.sav'

        #    classifier_model = pickle.load(open(filename, 'rb'))


        samples = embedding[0]
        
        svc_yhat_class = svc_model.predict(embedding)
        svc_yhat_prob = svc_model.predict_proba(embedding)


        yhat_class, yhat_prob = sim_predict(kmeans_model, samples, kmean_classes, top_n=5)
        
        end_time = time.time() - start_time
        print('Time for class predictions: Kmeans: {}'.format(end_time))


        class_probability = yhat_prob[0] * 100
        predict_names = yhat_class
        #print('Predicted: %s (%.3f)' % (predict_names, class_probability))
        compute_time = time.time() - overall_time

        res = {'svc_class': svc_classes[svc_yhat_class[0]], 'confidence': svc_yhat_prob[0, svc_yhat_class[0]]*100}

        print(f'face_rect is: {face_rect}')

        if enable_kmeans:
            res.update({'predictions': predict_names, 'probs': list(yhat_prob), 'compute_time': compute_time, 'face_rect': list(face_rect)})
        
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