import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import sys


import imageio
import time

#import resource
import subprocess
import os
import sys

number=17

source="SJCM00{}.mp4".format(number)
destination="C:/Temp/mono_depth/recognition/recognize_set_{}/".format(number)
dir_filenames="C:/Users/alexa/PycharmProjects/Semantic-Mono-Depth/utils/filenames"

window_points=[[19,118],[1261,493]]

# Устанавливаем разрешение
IM_WIDTH = 1280 #640
IM_HEIGHT = 720 #480

with open ("utils/filenames/recognition_{}.txt".format(number),"w"): #создаем пустой файл
    pass

# Выбор типа веб-камеры(if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')


# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

#filename = 'video/SJCM0019.mp4'
vid = imageio.get_reader(source, 'ffmpeg')

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    for num in range(1, 40000):#0000):
        print(str(num))
        num1 = num
        frame1 = vid.get_data(num)
        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.array(frame1)
        # for i in range(len(frame)):
        # if i>650:#85
        # for j in range(1280):
        # frame[i][j]=(255,255,255)

        # print(len(frame),len(frame[0]))
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        # Draw the results of the detection (aka 'visulaize the results')
        # print(len(frame),len(frame[0]))
        # print(str(num*100/2400)+" %")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame1 = frame[118:493,19:1261]


        # cv2.imwrite("result/"+str(num1)+".jpg",frame)
        cv2.imwrite(destination+str(num1).zfill(6)+"_10.jpg",frame1)
        with open ("utils/filenames/recognition_{}.txt".format(number),"a+") as file:
            file.write("recognize_set/"+str(num1).zfill(6)+"_10.jpg\n")
        # camera.close()

while (True):
    pass
cv2.destroyAllWindows()