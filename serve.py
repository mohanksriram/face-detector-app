
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
from tensorflow.keras.models import load_model
from flask import send_file
import pickle as p
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
import os
from tensorflow.keras.backend import set_session
import tensorflow as tf
import time

sess = tf.compat.v1.get_default_session()
graph = tf.get_default_graph()



# Utility functions
# extract a single face from a given photograph
def extract_face(image, required_size=(160, 160)):
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

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

def sim_predict(model, new_img_f, orig_classes, top_n=1, n_classes=5):
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

    classes = [orig_classes[val] for val in classes]
    
    #print(f'new_classes: {classes}')
    probs = sims[sims_top_n]
    #print(f'classes: {classes}')
    return classes, probs

app = Flask(__name__, static_folder="build/static",
 template_folder="build")

# All Model setup

set_session(sess)

# load the pre-trained Keras model
facenet_model = load_model('models/facenet_keras.h5')
#graph = tf.get_default_graph()
#facenet_model._make_predict_function()
filename = 'models/classifier_model.sav'
classifier_model = pickle.load(open(filename, 'rb'))
orig_classes = None
with open("models/orig_classes.txt", "rb") as fp:   # Unpickling
    orig_classes = pickle.load(fp)


@app.route("/")
def root():
    return render_template('index.html')
    
@app.route("/admin/dashboard")
def dashboard():
    return render_template('index.html')

@app.route("/logo")
def serve_img():
    filename = f'data/custom/train/disha/1.jpg'
    img = Image.open(filename)
    return serve_pil_image(filename)


@app.route('/serveImage/<path:path>')
def serve_latest(path):
    filename = f'data/custom/train/{path}/1.jpg'
    img = Image.open(filename)
    return serve_pil_image(filename)


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':

        start_time = time.time()
        
        print('received a get request')
        data_url = flask.request.args.get("url")
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(body)

        image = Image.open(BytesIO(body))
        
        # Face detection
        faces = extract_face(image)
        
        # Generate Embedding'
        # Load model if doesn't already exist
        #if not facenet_model:
        #    facenet_model = load_model('models/facenet_keras.h5')
        #with graph.as_default():
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            embedding = get_embedding(facenet_model, faces)

        in_encoder = Normalizer(norm='l2')
        embedding = np.expand_dims(embedding, axis=0)
        
        embedding = in_encoder.transform(embedding)

        # Perform prediction
        #if not classifier_model:
        #    filename = 'models/classifier_model.sav'

        #    classifier_model = pickle.load(open(filename, 'rb'))


        print(f'embedding shape: {embedding.shape}')
        samples = embedding[0]#np.expand_dims(embedding, axis=0)

        print(f'samples shape: {samples.shape}')

        #orig_classes = None
        #with open("models/orig_classes.txt", "rb") as fp:   # Unpickling
        #    orig_classes = pickle.load(fp)

        
        yhat_class, yhat_prob = sim_predict(classifier_model, samples, orig_classes, top_n=5)

        print(f'yhat class: {yhat_class}, yhat_prob: {yhat_prob}')

        #yhat_class = classifier_model.predict(samples)
        #yhat_prob = classifier_model.predict_proba(samples)

        # if not class_encoder:
        #     class_encoder = LabelEncoder()
        #     class_encoder.classes_ = np.load('models/classes.npy')




        # get name
        #class_index = yhat_class[0]
        class_probability = yhat_prob[0] * 100
        predict_names = yhat_class#class_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names, class_probability))
        # plot for fun
        # plt.imshow(faces)
        # title = '%s (%.3f)' % (predict_names[0], class_probability)
        # plot.title(title)
        # plot.show()
        
        compute_time = time.time() - start_time
        res = {'predictions': predict_names, 'probs': list(yhat_prob), 'compute_time': compute_time}
        print(res)
        return flask.jsonify(res)

    res = []
    return flask.jsonify(res)


def before_request():
    app.jinja_env.cache = {}

#print('Starting Flask!')

#app.debug=True
#app.run(host='0.0.0.0')


if __name__ == '__main__':
    int()
    port = os.environ.get('PORT', 5000)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, host='0.0.0.0', port=port)