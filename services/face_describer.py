from keras.models import load_model
from tensorflow.keras.backend import set_session
import tensorflow as tf
import numpy as np
import time

"""
Generate Embeddings for the face

"""

class FaceDescriber(object):

    def __init__(self, MODEL_PATH='models/facenet_keras.h5'):

        start_time = time.time()

        self.sess = tf.compat.v1.get_default_session()
        self.graph = tf.get_default_graph()

        set_session(self.sess)
        # load the pre-trained Keras model

        self.model = load_model(MODEL_PATH)

        end_time = time.time() - start_time
        print('Time for loading facenet model: {}'.format(end_time))


    def getEmbedding(self, face):

        with self.graph.as_default():
            set_session(self.sess)

            start_time = time.time()
            
            # scale pixel values
            face_pixels = face.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            # transform face into one sample
            samples = np.expand_dims(face_pixels, axis=0)
            # make prediction to get embedding

            yhat = self.model.predict(samples)
            
            end_time = time.time() - start_time
            print('Time for generating embedding: {}'.format(end_time))

            return yhat[0]