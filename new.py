import os
number=17


'''
filenames=os.listdir("C:/Temp/result_{}/".format(number))
print(filenames)
'''

source="C:/Temp/result_depth_17/"
destination="C:/Temp/result_colored_17/"

for k in range(6,88):
    dir_read=os.listdir(source+str(k))
    dir_read=sorted([int(l.replace(".png","")) for l in dir_read])
    dir_read=[str(l)+".png" for l in dir_read]
    print(dir_read)