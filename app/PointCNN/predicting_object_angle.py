'''Prepare Data for Classification Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/PointCNN")
import data_utils

import numpy as np
from tqdm import tqdm

import importlib
import tensorflow as tf

import glob
        

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(FLAG):
    
    data_filename = FLAG.filename
    point_indices = FLAG.point_indices
    
    print("point_indices", point_indices)
    
    
    CHECKPOINT_LOCATION = "/home/hasan/data/hdd8TB/KITTI/3D-object-detection/pointcnn-model/regression/pointcnn_cls_kitti_regression_2019-06-14-02-13-35_28595/iter-99500"
    
    
    # Root directory of the project

    # data_filename = FLAGS.image_filename
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    print(ROOT_DIR)

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Directory to get binary and image data
    PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
    DATA_DIR = os.path.join(PARENT_DIR, "test_dataset")

    drivename, fname = data_filename.split("/")

    bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".bin"

    
     ## Feed data to pointCNN
    
    xyzi = np.fromfile(
        os.path.join(bin_name),
        dtype=np.float32).reshape((-1, 4))
    
    
    data_feed = xyzi[point_indices]
    
    args_load_ckpt = CHECKPOINT_LOCATION
    args_model = "pointcnn_cls"
    args_setting = "kitti_regression"
    args_repeat_num = 1
    batch_size_val = 1
    
    model = importlib.import_module(args_model)
    setting_path = os.path.join(os.path.dirname(__file__), args_model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args_setting)

    sample_num = setting.sample_num
    step_val = setting.step_val
    rotation_range = setting.rotation_range
    rotation_range_val = setting.rotation_range_val
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val
    
    point_num = data_feed.shape[0]

    
    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, None,  data_feed.shape[1]), name='data_train')

    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if setting.use_extra_features:
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                features_augmented = features_sampled
    else:
        points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    net = model.Net(points=points_augmented, features=features_augmented, is_training=is_training, setting=setting)
    logits = net.logits
    probs = tf.nn.sigmoid(logits, name='probs')
    
    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    
    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args_load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args_load_ckpt))

        xforms_np, rotations_np = pf.get_xforms(batch_size_val,
                                                rotation_range=rotation_range_val,
                                                scaling_range=scaling_range_val,
                                                order=setting.rotation_order)
        
        _probs = sess.run([probs],
                     feed_dict={
                         pts_fts: data_feed,
                         indices: pf.get_indices(batch_size_val, sample_num, point_num,),
                         xforms: xforms_np,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter_val]),
                         is_training: False,
                     })
            
        print("_probs", _probs)
        
        
        np.array(_probs).tofile( FLAGS.output_file )

    return _probs


def _hash(_p):
    
    return str( round(_p[0], 4) )+str( round(_p[0], 4) )+str( round(_p[0], 4) )
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--point_indices',
        type=str,
        default='',
        help='point_indices.'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default='',
        help='filename.'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default='',
        help='output_file.'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
    
    print('{}-Done.'.format(datetime.now()))