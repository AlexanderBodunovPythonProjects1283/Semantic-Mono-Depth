import numpy as np
import cv2
import argparse
import sys
import os
sys.path.insert(0,"C:/Users/alexa/PycharmProjects/Semantic-Mono-Depth/utils")
import evaluation_utils
import time

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')
parser.add_argument('--mode',                type=str,   help='',                                     default="test")

args = parser.parse_args()


width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


if __name__ == '__main__':

    pred_disparities = np.load("C:/Temp/result/disparities_pp.npy")

    if args.split == 'kitti_test':
        if args.mode=='test':
            num_samples = 40
            gt_disparities = []
            ids = ['000150', '000106', '000174', '000032', '000127', '000001', '000064', '000134', '000003', '000039', '000175', '000033', '000087', '000129', '000160', '000072', '000093', '000167', '000178', '000161', '000089', '000105', '000067', '000035', '000138', '000193', '000125', '000128', '000004', '000048', '000038', '000123', '000111', '000042', '000184', '000185', '000116', '000119', '000095', '000019']
            for i in range(0,40):
                disp = cv2.imread(args.gt_path + "/training/disp_noc_0/" + ids[i] + "_10.png", -1)
                #print(args.gt_path + "\training\disp_noc_0\\" + ids[i] + "_10.png", -1)
                disp = disp.astype(np.float32) / 256
                gt_disparities.append(disp)
            gt_depths, pred_depths, pred_disparities_resized = evaluation_utils.convert_disps_to_depths_kitti(gt_disparities, pred_disparities)
        if args.mode == 'recognize':
            num_samples = 100
            gt_disparities = []

            pred_depths = []
            pred_disparities_resized = []

            for i in range(len(pred_disparities)):
                #gt_disp = pred_disparities[i]
                height, width = 375,1242

                pred_disp = pred_disparities[i]
                pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

                pred_disparities_resized.append(pred_disp)

                print(height,"   " ,width)
                width_to_focal_width=width_to_focal[width]
                pred_depth = width_to_focal_width * 0.54 / pred_disp
                pred_depths.append(pred_depth)


    if args.split == 'kitti':
        num_samples = 200
        
        gt_disparities = evaluation_utils.load_gt_disp_kitti(args.gt_path)
        gt_depths, pred_depths, pred_disparities_resized = evaluation_utils.convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

    elif args.split == 'eigen':
        num_samples = 697
        test_files = evaluation_utils.read_text_lines(args.gt_path + 'eigen_test_files.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = evaluation_utils.read_file_data(test_files, args.gt_path)

        num_test = len(im_files)
        gt_depths = []
        pred_depths = []
        for t_id in range(num_samples):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            depth = evaluation_utils.generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))

            disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
            disp_pred = disp_pred * disp_pred.shape[1]

            # need to convert from disparity to depth
            focal_length, baseline = evaluation_utils.get_focal_length_baseline(gt_calib[t_id], camera_id)
            depth_pred = (baseline * focal_length) / disp_pred
            depth_pred[np.isinf(depth_pred)] = 0

            pred_depths.append(depth_pred)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)


    k_=255
    k_3=k_**3
    k_2=k_**2

    count_=0

    for i in range(num_samples):
        if args.mode != 'recognize':
            count_+=1
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            if args.split == 'eigen':
                mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)


                if args.garg_crop or args.eigen_crop:
                    gt_height, gt_width = gt_depth.shape

                    # crop used by Garg ECCV16
                    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                    if args.garg_crop:
                        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                    # crop we found by trial and error to reproduce Eigen NIPS14 results
                    elif args.eigen_crop:
                        crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                         0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

            if args.split == 'kitti' or args.split == 'kitti_test':
                gt_disp = gt_disparities[i]
                mask = gt_disp > 0
                pred_disp = pred_disparities_resized[i]
                print(np.int32(pred_disp))
                print(gt_disp)
                print(mask)
                img_=np.uint8(pred_disp)

                max_=np.max(img_)
                avg_=int(np.sum(img_)/(len(img_)*len(img_[0])))

                print("max=",max_)
                print("avg=", avg_)
                #int(k_3/(k_2+i**2)),int(k_3/(k_2+(i-127)**2)),int(k_3/(k_2+(i-255)**2))
                img_1=np.array([[(int(255-0.4*i**2) if int(255-0.4*i**2)>0 else 0,int(255-0.06*(i-25)**2) if int(255-0.06*(i-25)**2)>0 else 0,int(255-0.06*(i-90)**2) if int(255-0.06*(i-90)**2)>0 else 0) for i in j] for j in img_])
                print(img_1)
                #for i in img_1:
                #    print (i)
                #img_=np.ones((100,100),np.uint8)*255
                #print(img_)
                #cv2.imshow("predicted_image",img_1)
                #if cv2.waitKey(1) == ord('q'):
                    #break
                cv2.imwrite("C:/Temp/result_depth/{}.png".format(count_),img_1)
                #time.sleep(5)
                disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
                bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
                d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = evaluation_utils.compute_errors(gt_depth[mask], pred_depth[mask])

        else:
            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth
            pred_disp = pred_disparities_resized[i]
            img_ = np.uint8(pred_disp)

            img_1=[]
            for i in img_:
                arr=[]
                for j in i:
                    b=int(255 - 0.4 * j ** 2)
                    g=int(255 - 0.06 * (j - 25) ** 2)
                    r=int(255 - 0.06 * (j - 90) ** 2)
                    arr.append((b if b>0 else 0,g if g>0 else 0,r if r>0 else 0))
                    img_1.append(arr)
            img_1=np.array(img_1)
            print("Картинка обработана .)")
            #img_1 = np.array([[(int(255 - 0.4 * i ** 2) if int(255 - 0.4 * i ** 2) > 0 else 0,int(255 - 0.06 * (i - 25) ** 2) if int(255 - 0.06 * (i - 25) ** 2) > 0 else 0,int(255 - 0.06 * (i - 90) ** 2) if int(255 - 0.06 * (i - 90) ** 2) > 0 else 0) for i in j] for j in img_])

            cv2.imwrite("C:/Temp/result_depth/{}.png".format(count_), img_1)


    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
