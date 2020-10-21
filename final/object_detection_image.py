######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# Name of the directory containing the object detection module we're using
IMAGE_NAME = '5.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})
true_box = boxes[0][scores[0] > 0.8]
print(true_box)
h, w = image.shape[1], image.shape[0]
ymin = int(true_box[0,0]*w)
xmin = int(true_box[0,1]*h)
ymax = int(true_box[0,2]*w)
xmax = int(true_box[0,3]*h)
image_crop = image[ymin : ymax, xmin : xmax]
# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image_crop)
cv2.imwrite("bienso5.JPG",image_crop)
#Press any key to close the image
cv2.waitKey(0)

# Clean up
#cv2.destroyAllWindows()

image_to_text = cv2.imread('bienso5.jpg')[:,:,0]
w, h =  image_to_text.shape
bigimage = np.ones((w*4, h*6), np.uint8)*255
bigimage[w*2:w*3, h*2:h*3] = 0
bigimage[w*2:w*3, h*2:h*3] = bigimage[w*2:w*3, h*2:h*3] + image_to_text
text = pytesseract.image_to_string(bigimage, lang = 'eng', config = '--psm 11')
print(text)
#cv2.imshow('im',bigimage)
#cv2.waitKey()
#cv2.destroyAllWindows()