import cv2 as cv
import os
import PIL
import matplotlib as plt
import matplotlib.pyplot as plt
from PIL import ImageOps
from PIL import Image
import numpy as np
import urllib.request
import os
import tarfile
import shutil
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm  #Progress bar
import scipy.misc

#Obtain the Model File
base_url = 'http://download.tensorflow.org/models/object_detection/'
file_name = 'ssd_mobilenet_v1_coco_2018_01_28.tar.gz'
url = base_url + file_name
urllib.request.urlretrieve(url, file_name)
os.listdir()

#Extract the Model Data
dir_name = file_name[0:-len('.tar.gz')]
if os.path.exists(dir_name):
  shutil.rmtree(dir_name) 
tarfile.open(file_name, 'r:gz').extractall('./')
os.listdir(dir_name)

labels = {}
with open('labels.txt', 'r') as labelsFile:
  for line in labelsFile.readlines():
    split = line.strip().split(': ')
    labels[int(split[0])] = split[1]
  print(labels)

#Input Video
input_video = cv.VideoCapture('cars.mp4')

height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
fps = input_video.get(cv.CAP_PROP_FPS)
total_frames = int(input_video.get(cv.CAP_PROP_FRAME_COUNT))

#Defining the output video
fourcc = cv.VideoWriter_fourcc(*'mp4v')
output_video = cv.VideoWriter('cars-detection.mp4', fourcc, fps, (300, 300))

#First Frame
input_video.set(cv.CAP_PROP_POS_FRAMES, 0)
ret, frame = input_video.read()

#Shape of image
frame_width_height = frame.shape

# Find the longer dimension 
max_dimension = max(frame_width_height)

# Compute the delta width and height
width_padding = max_dimension - frame_width_height[1]
height_padding = max_dimension - frame_width_height[0]

# Compute the padding amounts
left_padding = width_padding // 2
right_padding = width_padding - left_padding
top_padding = height_padding // 2
bottom_padding = height_padding - top_padding

# Pad and plot the image
padding = (left_padding,top_padding,right_padding,bottom_padding)

#Iterate through the the videos frames
#tqdm provides a progress bar for each iteration
#Reduce the video by grabbing the first frame of every second
for current_frame in tqdm(range(0, total_frames, int(fps))):
  input_video.set(cv.CAP_PROP_POS_FRAMES, current_frame)
  ret, frame_ = input_video.read()
  if not ret:
    raise Exception("Problem reading frame", i, " from video")

##################################################################
#   #Convert BGR to RGB
#   frame_= cv.cvtColor(frame_, cv.COLOR_BGR2RGB)

  #Convert the frame to an Image
  frame_ = Image.fromarray(frame_)

  #Apply padding to image
  padded_frame = ImageOps.expand(frame_, padding, (255,255,255,255))

  #Resize the image
  desired_size = (300, 300)
  resized_frame = padded_frame.resize(desired_size, Image.ANTIALIAS)
  
#######################################################################
  #Load the frozen graph
  frozen_graph = os.path.join(dir_name, 'frozen_inference_graph.pb')
  with tf.gfile.FastGFile(frozen_graph,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  #Convert the Image back to a numpy array
  image_np = np.asarray(resized_frame, dtype="int32")
  input_images = [image_np]

  #Shape of image
  width, height, _ = image_np.shape

  outputs = ('num_detections','detection_classes','detection_scores','detection_boxes',)

  #Run the graph
  with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    detections = sess.run([sess.graph.get_tensor_by_name(f'{op}:0') for op in outputs],
        feed_dict={ 'image_tensor:0': input_images })
  #Output
  num_detections = detections[0]
  detection_classes = detections[1]
  detection_scores = detections[2]
  detection_boxes = detections[3]

  #Number of object detected
  num_of_items = int(num_detections[0].item())

#######################################################################
  #Iterate through the number of object detected
  for i in range (num_of_items):

    #Label and Boundary features
    red = 255
    green = 255
    blue = 255
    scale = 0.5
    thickness = 2
    label = labels[detections[1][0][i]]

    #Object is a Car set color to blue
    if(detections[1][0][i] == 3):
      red = 0
      green = 0
      blue = 255

    #Object is a Truck set color to green
    if(detections[1][0][i] == 8):
      red = 0
      blue = 0
      green = 255

    #Dimensions of the Rectangle
    top = int(height*detections[3][0][i][0])
    left = int(width*detections[3][0][i][1])
    bottom = int(height*detections[3][0][i][2])
    right = int(width*detections[3][0][i][3])

    #Draw Rectangle
    cv.rectangle(image_np, (left, top), (right, bottom), (red, green, blue), thickness=2)
 
    #Draw Label
    cv.putText(image_np, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, scale, [red, green, blue], thickness)
    
  #making the video by outputting
  output_video.write(np.uint8(image_np))

  #Display image
  plt.imshow(image_np)
  plt.show()

input_video.release()
output_video.release()