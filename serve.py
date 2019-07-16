
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





#import cv2
facenet_model = None
classifier_model = None
class_encoder = None

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


app = Flask(__name__, static_folder="build/static",
 template_folder="build")

@app.route("/")
def hello():
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

        facenet_model = None
        classifier_model = None
        class_encoder = None
        
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
        if not facenet_model:
            facenet_model = load_model('models/facenet_keras.h5')

        embedding = get_embedding(facenet_model, faces)

        # Perform prediction
        if not classifier_model:
            filename = 'models/classifier_model.sav'

            classifier_model = pickle.load(open(filename, 'rb'))

        samples = np.expand_dims(embedding, axis=0)
        yhat_class = classifier_model.predict(samples)
        yhat_prob = classifier_model.predict_proba(samples)

        if not class_encoder:
            class_encoder = LabelEncoder()
            class_encoder.classes_ = np.load('models/classes.npy')


        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = class_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        # plot for fun
        # plt.imshow(faces)
        # title = '%s (%.3f)' % (predict_names[0], class_probability)
        # plot.title(title)
        # plot.show()
        filename = f'data/custom/train/{predict_names[0]}/1.jpg'
        
        res = {'prediction_name': predict_names[0], 'prob': class_probability}
        print(res)
        return flask.jsonify(res)
        #return send_file(filename, mimetype='image/jpg')

        w=10
        h=10
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 1, 1)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        plt.title(title)
        plt.imshow(faces)

        img = Image.open(f'data/custom/train/{predict_names[0]}/1.jpg')
        fig.add_subplot(2, 1, 2)
        plt.imshow(img)
        plt.show()


        
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(f'final decoded image!')
    else:
        print("received a new post request!")
        print(f'flask request is: {flask.request}')
        print(f'flask request values are: {flask.request.values}')
        data_url = flask.request.values['imageBase64']
        #print(f"flask request files are {flask.request.files}")
        #bytes = flask.request.files['file'].read()
        #img = load_image_bytes(bytes)
    res = []#predict(img)
    return flask.jsonify(res)



print('Starting Flask!')

app.debug=True
app.run(host='0.0.0.0')