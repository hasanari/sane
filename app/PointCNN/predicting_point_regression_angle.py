'''Prepare Data for Semantic3D Segmentation Task.'''

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
    
    filtered_point_indices = FLAG.indices
    data_filename = FLAG.filename
    ground_removed = FLAG.ground_removed
    retrieve_whole_files = FLAG.retrieve_whole_files
    #CHECKPOINT_LOCATION = "/home/hasan/data/hdd8TB/KITTI/3D-object-detection/pointcnn-model/random-XYZ-grid-0.25/pointcnn_seg_kitti3d_x8_2048_fps_2019-04-09-19-00-45_10534/ckpt-best/0.9259112-iter--150000"
    
    
    CHECKPOINT_LOCATION = "/home/hasan/data/hdd8TB/KITTI/3D-object-detection/pointcnn-model/pointcnn_bin_based_regression/enlarge-1.0/car-only+dim-4+Fc-head-Yes+Head-Bin-9+sample-512+DP-0.1/pointcnn_bin_based_regression_kitti_regression_2019-06-20-17-14-40_30733/ckpts-best/36.04293-iter--41500"
    
    max_point_num = 2048

    
    
    # Root directory of the project

    # data_filename = FLAGS.image_filename
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    print(ROOT_DIR)

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Directory to get binary and image data
    PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
    DATA_DIR = os.path.join(PARENT_DIR, "test_dataset")
    
    
     ## Feed data to pointCNN
    
    args_load_ckpt = CHECKPOINT_LOCATION
    args_model = "pointcnn_bin_based_regression"
    args_setting = "kitti_regression"
    args_repeat_num = 1
    
    model = importlib.import_module(args_model)
    setting_path = os.path.join(os.path.dirname(__file__), args_model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args_setting)

    sample_num = setting.sample_num
    batch_size = args_repeat_num * int ( math.ceil(max_point_num / sample_num) )

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    net = model.Net(points_sampled, features_sampled, is_training, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    
    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args_load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args_load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))



        drivename, fname = data_filename.split("/")
        
        all_files = [fname]
        if(retrieve_whole_files):
            
            if(ground_removed == 1):
                
                source_path = os.path.join(DATA_DIR, drivename, "ground_removed" )
            else:
                
                source_path = os.path.join(DATA_DIR, drivename, "bin_data" )
        
            glob_files = (glob.glob(source_path+"/*.bin"))

            all_files = [fname] + [ _i.replace(source_path+"/", "")[0:-4] for _i in glob_files]
            
        for fname in all_files:
            
            if(ground_removed == 1):
                
                output_location = os.path.join(ROOT_DIR, "output/"+drivename+"_"+fname+"ground_removed.bin")
            else:
                
                output_location = os.path.join(ROOT_DIR, "output/"+drivename+"_"+fname+FLAGS.postfix+".bin")
            
            if os.path.isfile( output_location ) :
                continue

            label_length, data, data_num, indices_split_to_full, item_num, all_label_pred, indices_for_prediction = data_preprocessing(drivename, fname , max_point_num, ground_removed)


            merged_label_zero = np.zeros((label_length),dtype=int)
            merged_confidence_zero = np.zeros((label_length),dtype=float)


            data =data[0:item_num, ...].astype(np.float32) 

            data_num =data_num[0:item_num, ...] 
            indices_split_to_full = indices_split_to_full[0:item_num]

            batch_num = data.shape[0]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)


            for batch_idx in range(batch_num):
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = int ( math.ceil((sample_num * batch_size) / point_num) )
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)


                seg_probs = sess.run([seg_probs_op],
                                        feed_dict={
                                            pts_fts: points_batch,
                                            indices: indices_batch.astype(np.int32),
                                            is_training: False,
                                        })
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])


            for idx in range(batch_num): #Get highest confidence
                pred = labels_pred[idx].astype(np.int64)
                _indices = indices_split_to_full[idx].astype(np.int64)
                confidence = confidences_pred[idx].astype(np.float32)
                num = data_num[idx].astype(np.int64)

                for i in range(pred.shape[0]):

                     if confidence[i] > 0.8 and confidence[i] > merged_confidence_zero[_indices[i]]:
                        merged_confidence_zero[_indices[i]] = confidence[i]
                        merged_label_zero[_indices[i]] = pred[i]


            all_label_pred[indices_for_prediction] = merged_label_zero



            #print("all_label_pred", np.unique(all_label_pred, return_counts=True))
            
            
            bounded_indices = ( all_label_pred == 2 ).nonzero()[0]


            bounded_indices.tofile(output_location)
            
            
            #print('{}-Done!'.format(datetime.now()), output_location)
    
    return

    #Remove Ground points
    """
    fh = FrameHandler()
    bp = BoundingBoxPredictor(fh)
    
    pc = xyzi[bounded_indices]
    
    hash_points = {}
    for _idx in bounded_indices:
        hash_points[_hash(xyzi[_idx, 0:3])] = _idx
    
    png = bp.ground_plane_fitting(pc[:, 0:3])["png"]
    
    #print(hash_points)
    bounded_indices = []
    for _point in png:
        
        bounded_indices.append(hash_points[_hash(_point[0:3])])
        
    #print(bounded_indices)
    bounded_indices = np.array(bounded_indices)
    bounded_indices.tofile(os.path.join(ROOT_DIR, "output/"+drivename+"_"+fname+".bin"))
    
    #bounded_indices.tofile(os.path.join(ROOT_DIR, "output/indices.bin"))
    
    """
def _hash(_p):
    
    return str( round(_p[0], 4) )+str( round(_p[0], 4) )+str( round(_p[0], 4) )
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--postfix',
        type=str,
        default='',
        help='postfix.'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default='',
        help='filename.'
    )
    
    parser.add_argument(
        '--ground_removed',
        type=int,
        default='0',
        help='ground_removed.'
    )
    
    
    parser.add_argument(
        '--retrieve_whole_files',
        type=int,
        default='0',
        help='retrieve_whole_files.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
    
    print('{}-Done.'.format(datetime.now()))