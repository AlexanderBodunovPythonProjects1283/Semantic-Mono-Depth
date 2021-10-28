import numpy as np
import cv2
import argparse
import sys
import os

sys.path.insert(0, "C:/Users/alexa/PycharmProjects/Semantic-Mono-Depth/utils")
import evaluation_utils
import time

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split', type=str, help='data split, kitti or eigen', required=True)
parser.add_argument('--predicted_disp_path', type=str, help='path to estimated disparities', required=True)
parser.add_argument('--gt_path', type=str, help='path to ground truth disparities', required=True)
parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=180)
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--mode', type=str, help='', default="test")
parser.add_argument('--result_folder', type=str, help='',)

args = parser.parse_args()

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351

number=19

if __name__ == '__main__':
    result_folder="C:/Temp/result_depth_{}/".format(number)
    #filenames=os.listdir("C:/Temp/result_{}/".format(number))
    #num_steps=len(filenames)
    count_ =0
    for j in range (6,199):
        path_result=result_folder+str(j)+"/"
        try:
            os.mkdir(path_result)
        except:
            pass
        pred_disparities = np.load("C:/Temp/result_{0}/disparities_{1}.npy".format(number,j))
        num_samples = 100
        gt_disparities = []

        pred_depths = []
        pred_disparities_resized = []

        for i in range(len(pred_disparities)):

            # gt_disp = pred_disparities[i]
            height, width = 375, 1242

            pred_disp = pred_disparities[i]
            pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

            pred_disparities_resized.append(pred_disp)

            print(height, "   ", width)
            width_to_focal_width = width_to_focal[width]
            pred_depth = width_to_focal_width * 0.54 / pred_disp
            pred_depths.append(pred_depth)


        for i in range(num_samples):
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth
            pred_disp = pred_disparities_resized[i]
            img_ = np.uint8(pred_disp)

            print("Картинка обработана .)")
            # img_1 = np.array([[(int(255 - 0.4 * i ** 2) if int(255 - 0.4 * i ** 2) > 0 else 0,int(255 - 0.06 * (i - 25) ** 2) if int(255 - 0.06 * (i - 25) ** 2) > 0 else 0,int(255 - 0.06 * (i - 90) ** 2) if int(255 - 0.06 * (i - 90) ** 2) > 0 else 0) for i in j] for j in img_])

            cv2.imwrite(path_result+str(count_)+".png", img_)
            count_+=1