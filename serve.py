
import flask
from flask import Flask, render_template
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import requests
import base64
#import cv2



def load_image_url(url: str):
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString):
    img = open_image(BytesIO(raw_bytes))
    return img


app = Flask(__name__, static_folder="build/static",
 template_folder="build")

@app.route("/")
def hello():
    return render_template('index.html')
    
@app.route("/admin/dashboard")
def dashboard():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        print('received a get request')
        data_url = flask.request.args.get("url")
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(body)
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