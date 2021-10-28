import numpy as np
import cv2
import os
import imageio
import time
import math

number=19

filename='C:/Users/alexa/PycharmProjects/self-driving_car_objects_recognition/video/SJCM00{}.mp4'.format(number)
vid=imageio.get_reader(filename,'ffmpeg')

source="C:/Temp/result_depth_{}/".format(number)
destination="C:/Temp/result_colored_{}/".format(number)

count_=0
pow_=1.9



def grater_0(num):
    if num>0:
        return(num)
    else:
        return 0


for k in range(6,200):
    dir_read=source+str(k)+"/"
    files=os.listdir(dir_read)
    files=sorted([int(l.replace(".png","")) for l in files])
    files=[str(l)+".png" for l in files]

    for l in range(len(files)):
        count_+=1
        img_original=vid.get_data(count_)
        img_original=cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB)
        img_=cv2.imread(dir_read+files[l])
        img_=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        #img_1 = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
        print(dir_read+str(l))
        img_b=np.copy(img_)
        img_g = np.copy(img_)
        img_r = np.copy(img_)
        #img_b[img_b >25] = 25.24

        img_b=(255 - 0.4 * (img_b ** pow_))

        #img_g[img_g > 90] = 90.19
        #img_g[img_g < 25] = 90.19
        img_g = (255 - 0.06 * (img_g - 25) ** pow_)

        #img_r[img_r < 155]=155.19
        img_r=(255 - 0.06 * (img_r - 90) ** pow_)
        img_1=np.zeros((375,1242,3))
        img_1[:,:,0]=img_b
        img_1[:,:,1]=img_g
        img_1[:,:,2]=img_r

        img_original[118: 493, 50: 1230]=img_1[:, 31: 1211]
        cv2.imwrite(destination+str(count_)+".png",img_original)
        print(count_)

#Показваем последнее изображение

#cv2.imshow("uuu",img_original)
#if cv2.waitKey(1) == ord('q'):
    #break