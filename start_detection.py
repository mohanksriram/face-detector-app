import cv2
RTSP_URL = 'rtsp://admin:admin@192.168.1.117:554'

def start_detection(url):
    print('starting detection')
    vidcap = cv2.VideoCapture(url)
    success, image = vidcap.read()
    count = 0
    while success:
        #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success, image = vidcap.read()
        cv2.imshow('VIDEO', image)
        cv2.waitKey(1)
        #print('Read a new frame: ', success)
        count += 1

if __name__ == "__main__":
    start_detection(RTSP_URL)