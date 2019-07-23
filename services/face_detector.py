import cv2
import time
import numpy as np
from PIL import Image  #TODO: replace with opencv
from pathlib import Path

class FaceDetector(object):

    def __init__(self, down_scale_factor=1.0):
        self.down_scale_factor = down_scale_factor

    def _isWithinThreshold(self, image, bbox, threshold=20):

        print(f'bbox: {bbox}')
        x1, y1, x2, y2 = bbox

        print(f'face box: {bbox}, width: {image.shape[0]}, height: {image.shape[1]}')


        if x1 < threshold or image.shape[1] - x2 < threshold or y1 < threshold or image.shape[0] - y2 < threshold: 
            return False

        return True

    def getFacelocations(self, image, model_file, config_file):
        
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        frame_width = image.shape[1]
        frame_height = image.shape[0]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                area = width * height

                bboxes.append(([x1, y1, x2, y2], area))
        bboxes.sort(key=lambda x: x[1], reverse=True)

        return bboxes

    
    def getFaces(self, image, required_size=(160, 160)):

        model_file = "opencv_face_detector_uint8.pb"
        config_file = "opencv_face_detector.pbtxt"
        dnn_weights_path = 'models'
        model_file = Path(f'{dnn_weights_path}/{model_file}')
        config_file = Path(f'{dnn_weights_path}/{config_file}')

        print(f'model file: {model_file}, config file: {config_file}')

        start_time = time.time()
        
        image = np.array(image, dtype='uint8')
        face_locations = self.getFacelocations(image, model_file, config_file)

        print(f'face locations: {face_locations}')

        end_time = time.time() - start_time
        print('Time for dnn Detector: {}'.format(end_time))

        if len(face_locations) > 0 and self._isWithinThreshold(image=image, bbox=face_locations[0][0]):

            x1, y1, x2, y2 = face_locations[0][0]
            face_rect = face_locations[0][0]
            
            full_pixels = image

            # face = full_pixels[top:bottom, left:right]
            face = full_pixels[y1:y2, x1:x2]

            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = np.asarray(image)
            
            return face_array, face_rect

        else:
            return [], []

    

