import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 70
FONT = cv2.FONT_HERSHEY_DUPLEX


def get_rps_prediction(arr):
    m = arr[0][0]
    midx = 0
    for i in range(len(arr[0])):
        if m < arr[0][i]:
            m = arr[0][i]
            midx = i
    if midx == 0:
        return 'ROCK'
    elif midx == 1:
        return 'PAPER'
    else:
        return 'SCISSORS'


def scale(path):
    image_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model('CNN-rps.nn')


cam = cv2.VideoCapture(0)


if not cam.isOpened():
  print ("Error: Couldn't open cam")
  exit()

while(1):
    ret, frame = cam.read()
    if ret:
        frame = cv2.flip(frame,1)
        display = cv2.rectangle(frame.copy(),(200,100),(500,400),(0,255,0),5) 
        cv2.imshow('curFrame',display)
        ROI = frame[100:400, 200:500].copy()
        cv2.imshow('Current Roi', ROI)
        cv2.imwrite("ROI.png",ROI)
        img = scale('ROI.png')
        prediction = model.predict([img])
        cv2.putText(frame,f'{get_rps_prediction(prediction)}',(50,50), FONT, 2 ,(255,255,255), 2, cv2.LINE_8)
        cv2.imshow('curFrame', frame) 


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()