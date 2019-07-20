
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
from tensorflow.keras.backend import set_session
import tensorflow as tf
import time
import face_recognition

sess = tf.compat.v1.get_default_session()
graph = tf.get_default_graph()

import logging

logging.basicConfig(filename="Log_Test_File.txt",
                level=logging.DEBUG,
                format='%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S')


enable_kmeans = True


# Utility functions
# extract a single face from a given photograph
def extract_face(image, required_size=(160, 160)):
    # convert to RGB, if needed
    image = image.convert('RGB')
    #image.setFlags(write=True)
    # convert to array
    #pixels = asarray(image)

    resize_factor = 2

    small_frame = image.resize((int(image.width * 1/resize_factor), int(image.height * 1/resize_factor)))

    pixels = np.array(small_frame)
    # create the detector, using default weights
    start_time = time.time()
    # detector = MTCNN()


    # # detect faces in the image
    # results = detector.detect_faces(pixels)


    # # extract the bounding box from the first face
    # x1, y1, width, height = results[0]['box']
    # # bug fix
    # x1, y1 = abs(x1), abs(y1)
    # x2, y2 = x1 + width, y1 + height

    face_locations = face_recognition.face_locations(pixels, number_of_times_to_upsample=0, model="cnn")

    end_time = time.time() - start_time
    logging.info('Time for MTCNN Detector: {}'.format(end_time))
    print('Time for face_recognition Detector: {}'.format(end_time))

    #print(f'image shape: {pixels.shape}')
    # Print the location of each face in this image

    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        face_rect = face_locations[0]

        top *= resize_factor
        right *= resize_factor
        bottom *= resize_factor
        left *= resize_factor
        #print(f'face_location: {face_locations[0]}')
        # extract the face
        full_pixels = np.array(image)
        face = full_pixels[top:bottom, left:right]
        #print(f'resized image: {face.shape}')
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

    else:
        return [], []
    
    return face_array, face_rect

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding

	yhat = model.predict(samples)
	return yhat[0]

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

# All Model setup

set_session(sess)

# load the pre-trained Keras model
start_time = time.time()

facenet_model = load_model('models/facenet_keras.h5')

end_time = time.time() - start_time
logging.info('Time for loading facenet model: {}'.format(end_time))
print('Time for loading facenet model: {}'.format(end_time))


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
logging.info('Time for loading classifier: {}'.format(end_time))
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


@app.route('/serveImage/<path:path>')
def serve_latest(path):
    filename = f'data/custom/train/{path}/image_0.png'
    img = Image.open(filename)
    return serve_pil_image(filename)


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        overall_time = time.time()
        start_time = time.time()
        
       # print('received a get request')
        data_url = flask.request.args.get("url")
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))

        end_time = time.time() - start_time
        logging.info('Time for Decoding image: {}'.format(end_time))
        print('Time for Decoding image: {}'.format(end_time))

        image = Image.open(BytesIO(body))
        
        # Face detection
        faces, face_rect = extract_face(image)
        
        if len(faces) == 0:
            compute_time = time.time() - overall_time
            prediction_names = ['nobody']
            res = {'svc_class': prediction_names[0], 'confidence': 100}
            res.update({'predictions': prediction_names, 'probs': [1], 'compute_time': compute_time, 'face_rect': [0, 0, 0, 0]})
            return flask.jsonify(res)



        # Generate Embedding'
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            start_time = time.time()
            embedding = get_embedding(facenet_model, faces)
            end_time = time.time() - start_time
            logging.info('Time for generating embedding : {}'.format(end_time))
            print('Time for generating embedding: {}'.format(end_time))



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
        logging.info('Time for class predictions: Kmeans: {}'.format(end_time))
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