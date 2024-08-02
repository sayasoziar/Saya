import os , random , cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras , pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


#green color for rectangle of object detection
green = (0 , 255 , 0)
#light blue color for face detection
light_blue =  (255 , 125 , 125)

mask_detection =tf.keras.models.load_model('model.h5')

text_mask = "Mask on"
text_no_mask = "Mask off"

font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.8

# then this is our face detection file that we can detect faces
# which is one of the cascade classfires you can download in internet
face_detection = cv2.CascadeClassifier('E:/Pycharm/module/haarcascade_frontalface_default.xml')
# then we need our two files for object detection which are mobile ssd and froxen file (weight)
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weight_file = 'frozen_inference_graph.pb'
# these to files should be on the same folder of project file(python) otherwise it will give you error
cam = cv2.VideoCapture(0)# to access camera of our device and 0 meas the deafult camera
                         # some times we may have multi camera and this is the purpose of it
# and vy those two files that we imported in out project we combine them and make detection model
net = cv2.dnn_DetectionModel(weight_file , config_file) # our detection model
net.setInputSize(320 , 320) # required size of our detection model
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5 , 127.5 , 127.5))
net.setInputSwapRB(True)
# these codes that written above they are required because the detection model is not,
# our trained model ,So we have to write down them

# then our classes comes ,name of our objected that trained to detect which are about 90
class_file = 'coco.txt'
class_names = [] # an empty array to add our object names to it by the following action

with open(class_file , 'rt') as fpt:
     class_names = fpt.read().rstrip('\n').split('\n')
    # so this process based on split and strip on every new lin
def predict(image):
    face_frame = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    # to resize for comman size deep learning models
    face_frame = cv2.resize(face_frame , (224 ,224))
    # converting to numpy array
    face_frame = img_to_array(face_frame)
    # the value is 3 chanells and it expand one more chanell for batch file
    # it going to be like that (1 , 224 , 224 , 3)
    face_frame = np.expand_dims(face_frame , axis = 0)
    # pre processing for the deep learning and makee every thing is perfect
    face_frame = preprocess_input(face_frame)
    # finally put it in to the deep learning file
    prediction = mask_detection.predict(face_frame)

    return prediction[0][0]

# a function to do our process and reduce codes in purpose of clearing our codes
def display_fun(cam):
    cam_gray = cv2.cvtColor(cam , cv2.COLOR_BGR2GRAY)
    # in case cade (face detection file) works on gray images that is why we convert it to gray
    box  = face_detection.detectMultiScale(cam , minNeighbors = 8 )
    # box is the face bounder that is got detect (x1 , y1) , (x2 , y2)
    print(box) # just for seeing boundery numbers
    for x1 , y1 , width , height in box:
         x2 , y2 = x1 + width , y1 + height
          # for creating rectangle around the face we need to have 4 points 2 of them for strart
          # and the another two points for wnding that is why we get it by the height and width of the face
         cv2.rectangle(img = cam , pt1 = (x1 -10 , y1 -10) , pt2 = (x2 +10 , y2 + 10) , color = light_blue , thickness = 2)
         # and we used those optional keywords(img , pt1 , pt2 , etc.) for make it more understandable
         # this is the rectangle shaping on the rgb image which is mean get values of the rectangle
         # from gray images and put it them on a rgb image to show it

    for(x , y , w , h) in box:
        roi_color = cam[y : y+h , x: x+w]
        mask = predict(roi_color)

        if mask > 0.45:
            cv2.rectangle(cam , (x,y ) , (x+w , y+h)  , (128 , 0 , 128) , 2)
            cv2.putText(cam , text = text_mask , org = (x+50 , (y+w)-10) , fontFace= font , fontScale= scale , color= (128 , 0,  128) , thickness=2)
        elif mask <= 0.5:
            cv2.rectangle(cam, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(cam, text=text_no_mask, org=(x + 50, (y+w)-10), fontFace=font, fontScale=scale, color=(0, 0, 255),
                        thickness=2)
    return(cam)

while True: # starting of real time shoing
    ret , frame_img = cam.read() # unpacking each frame and add it in the variable of frame_img
    class_id , confs , boundary_box = net.detect(frame_img , confThreshold = 0.5)
    # class_id = (index_num - 1 ) of the object names
    # confs the accuracy of object detection
    # boundary_box , the object detection points that we start with it and end it just as the rectangle
    # and confs mini is equal to 0.5 this code means the accuracy of the object must greater than 0.5
    # then the program consider as an object
    print(boundary_box)
    # just to see the bounder numbers
        # getting value of those variables to our (for) and ,they are more than one that is why we use zip
    for classID, confidence, box in zip(class_id, confs, boundary_box):
        cv2.rectangle(frame_img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(frame_img, class_names[classID -   1].upper(), (box[0] + 10, box[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color =(0 , 255 ,0), thickness=2)
        face_display = display_fun(frame_img)
        cv2.imshow('window', face_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()