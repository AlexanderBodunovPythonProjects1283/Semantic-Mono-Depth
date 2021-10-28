import cv2
import numpy as np

from monodepth_model import *
from monodepth_dataloader import *
from monodepth_my_set_loader import *
from average_gradients import *
from . import utils_
import os


#video_input="C:/Users/alexa/PycharmProjects/self-driving_car_objects_recognition/video/SJCM0017.mp4"
video_input="SJCM0017.mp4"


output_dir="C:/Temp/mono_depth/output_disparity/"

height, width = 375, 1242

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


min_depth=1e-3
max_depth=180

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)



def recognize():
    dataset="kitti"
    mode="recognize"
    data_path="C:/Temp/mono_depth/recognition"
    output_dir="C:/Temp/mono_depth/result"
    filename="C:/Users/alexa/PycharmProjects/Semantic-Mono-Depth/utils/filenames/recognition.txt"
    task="depth"
    checkpoint_path="C:/Temp/mono_depth/checkpoints/vgg/model-16000"
    encoder="vgg"
    log_directory="'./logs/'"
    model_name='semantic-monodepth'
    height=256
    width=600
    batch_size=2
    num_threads=8
    num_epochs=50
    do_sterio='store_true'
    wrap_mode='border'
    use_deconv='store_true'
    alpha_image_loss=0.85
    disp_gradient_loss_weight=0.1
    lr_loss_weight=1.0
    full_summary='store_true'

    params = monodepth_parameters(
        encoder=encoder,
        height=height,
        width=width,
        batch_size=batch_size,
        num_threads=num_threads,
        num_epochs=num_epochs,
        do_stereo=do_sterio,
        wrap_mode=wrap_mode,
        use_deconv=use_deconv,
        alpha_image_loss=alpha_image_loss,
        disp_gradient_loss_weight=disp_gradient_loss_weight,
        lr_loss_weight=lr_loss_weight,
        task=task,
        full_summary=full_summary)


    """Test function."""

    #dataloader = MonodepthDataloaderMy("C:/Temp/mono_depth/recognition", filename, params,"kitti","recognize")
    dataloader = MonodepthDataloaderMy(data_path, filename, params,dataset,mode)
    dataset = dataloader.left_image_batch
    semantic=valid=vars_to_restore= []  # dataloader.right_image_batch

    print("_"*100,"dataset\n",dataset,"_"*100)

    #model = MonodepthModel(params, "recognize", "depth", MonodepthDataloaderMy.left_image_batch, MonodepthDataloaderMy.left_image_batch, [], [])
    model = MonodepthModel(params, mode, task, dataset, dataset, semantic, valid)

    if checkpoint_path != '' and len(vars_to_restore) == 0:
        vars_to_restore = utils_.get_var_to_restore_list(checkpoint_path)
        print('Vars to restore ' + str(len(vars_to_restore)) + ' vs total vars ' + str(len(tf.trainable_variables())))
    else:
        print("No vars :(")

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_loader = tf.train.Saver()
    if checkpoint_path != '':
        train_loader = tf.train.Saver(var_list=vars_to_restore)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(log_directory + '/' + model_name)
    else:
        restore_path = checkpoint_path
    train_loader.restore(sess, restore_path)

    num_test_samples = count_text_lines(filename)
    # num_test_samples=100

    print('now testing {} files'.format(num_test_samples))


    loop_size = int(num_test_samples / 100)
    for count in range(6,loop_size):
        disparities = np.zeros((100, height, width), dtype=np.float32)
        disparities_pp = np.zeros((100, height, width), dtype=np.float32)
        print("Память была выделена...")
        for step in range(100):
            #step_ = count * 100 + step
            disp = sess.run(model.disp_left_est[0])
            disparities[step] = disp[0].squeeze()
            disparities_pp[step] = post_process_disparity(disp.squeeze())
        progress = count / loop_size * 100
        print("Обработано {0} процентов файлов".format(progress))

        print('done.')

        print('writing results.')
        if output_directory == '':
            output_directory = os.path.dirname(checkpoint_path)
        else:
            output_directory = output_directory


        #np.save('{0}/disparities_{1}.npy'.format(output_directory,count), disparities)
        #np.save('{0}/disparities_pp_{1}.npy'.format(output_directory,count), disparities_pp)


    print('done.')


recognize()

inp = cv2.VideoCapture(video_input)
video_width = int(inp.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = inp.get(cv2.CAP_PROP_FPS)

print('Video resolution: (' + str(video_width) + ', ' + str(video_height) + ')')
print('Video fps:', video_fps)

print('Video is running')
info = []


'''

frame_id = 0
while inp.isOpened() and frame_id<100:
        ret, frame = inp.read()
        #try:
        #if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
            #break

        frame.setflags(write=1)
        #frame_expanded = np.expand_dims(frame, axis=0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame1 = frame[118:493, 19:1261]

        pred_disp=frame1
        pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)

        width_to_focal_width = width_to_focal[width]
        pred_depth = width_to_focal_width * 0.54 / pred_disp


        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        img_ = np.uint8(pred_disp)
        cv2.imwrite(output_dir + str(frame_id) + ".png", pred_depth)
        frame_id+=1

        print(frame_id)
'''