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


def data_preprocessing(drivename, fname , max_point_num, ground_removed, FLAGS):
    batch_size = 2048
    block_size = 1000
    grid_size = 0.25
        
    # Root directory of the project

    # data_filename = FLAGS.image_filename
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Directory to get binary and image data
    PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
    DATA_DIR = os.path.join(PARENT_DIR, "test_dataset")
    
    if(ground_removed == 1):
        
        bin_name = os.path.join(DATA_DIR, drivename, "ground_removed", fname) + ".bin"
        
        if(os.path.isfile(bin_name) == False ):
            bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".bin"
            
    else:
        bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".bin"
    
    #print(ground_removed, bin_name)
    
    data = np.zeros((batch_size, max_point_num, 4))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)


    
    xyzi = np.fromfile(
        os.path.join(bin_name),
        dtype=np.float32).reshape((-1, 4))
    
    
    indices_for_prediction = np.arange(xyzi.shape[0]) #(xyzi[:,0] >= -5 ).nonzero()[0]
    #print("indices_for_prediction", indices_for_prediction)
    # Filter point only in front on of ego-sensors
    xyzif =xyzi #= xyzi[xyzi[:,0] >= -5 ] 
    
    all_label_pred = np.zeros((xyzi.shape[0]),dtype=int)
    label_length = xyzif.shape[0]
    xyz =xyzif[:,0:3]
    
    
    if(FLAGS.postfix == "denoise-weights"):        
        i =( xyzif[:,3:4] / (np.max(xyzif[:,3:4].flatten()) + 1e-10) ) - 0.5

    elif(FLAGS.postfix == "normal-weights"):
        i =( xyzif[:,3:4] / (np.max(xyzif[:,3:4].flatten()) + 1e-10)  ) 
    
            
            
            
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    block_size = (2 * (xyz_max[0, 0] - xyz_min[0, 0]), 2 * (xyz_max[0, 1] - xyz_min[0, 1]) ,  2 * (xyz_max[0, -1] - xyz_min[0, -1]))
    
    xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

    #print('{}-Collecting points belong to each block...'.format(datetime.now(), xyzrcof.shape[0]))
    blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                return_counts=True, axis=0)
    block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
    #print('{}-{} is split into {} blocks.'.format(datetime.now(), dataset, blocks.shape[0]))

    block_to_block_idx_map = dict()
    for block_idx in range(blocks.shape[0]):
        block = (blocks[block_idx][0], blocks[block_idx][1])
        block_to_block_idx_map[(block[0], block[1])] = block_idx

    # merge small blocks into one of their big neighbors
    block_point_count_threshold = max_point_num / 3
    #print("block_point_count_threshold",block_point_count_threshold)
    nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    block_merge_count = 0
    for block_idx in range(blocks.shape[0]):
        if block_point_counts[block_idx] >= block_point_count_threshold:
            #print(block_idx, block_point_counts[block_idx])

            continue


        block = (blocks[block_idx][0], blocks[block_idx][1])
        for x, y in nbr_block_offsets:
            nbr_block = (block[0] + x, block[1] + y)
            if nbr_block not in block_to_block_idx_map:
                continue

            nbr_block_idx = block_to_block_idx_map[nbr_block]
            if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                continue


            #print(block_idx, nbr_block_idx, block_point_counts[nbr_block_idx])

            block_point_indices[nbr_block_idx] = np.concatenate(
                [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
            block_point_indices[block_idx] = np.array([], dtype=np.int)
            block_merge_count = block_merge_count + 1
            break
    #print('{}-{} of {} blocks are merged.'.format(datetime.now(), block_merge_count, blocks.shape[0]))

    idx_last_non_empty_block = 0
    for block_idx in reversed(range(blocks.shape[0])):
        if block_point_indices[block_idx].shape[0] != 0:
            idx_last_non_empty_block = block_idx
            break

    # uniformly sample each block
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        #print(block_idx, point_indices.shape)
        block_points = xyz[point_indices]
        block_min = np.amin(block_points, axis=0, keepdims=True)
        xyz_grids = np.floor((block_points - block_min) / grid_size).astype(np.int)
        grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                 return_counts=True, axis=0)
        grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
        grid_point_count_avg = int(np.average(grid_point_counts))
        point_indices_repeated = []
        for grid_idx in range(grids.shape[0]):
            point_indices_in_block = grid_point_indices[grid_idx]
            repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
            if repeat_num > 1:
                point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                np.random.shuffle(point_indices_in_block)
                point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
            point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
        block_point_indices[block_idx] = np.array(point_indices_repeated)
        block_point_counts[block_idx] = len(point_indices_repeated)

    idx = 0
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        block_point_num = point_indices.shape[0]
        block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))
        point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
        point_nums = [point_num_avg] * block_split_num
        point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
        starts = [0] + list(np.cumsum(point_nums))

        np.random.shuffle(point_indices)
        block_points = xyz[point_indices]


        block_min = np.amin(block_points, axis=0, keepdims=True)
        block_max = np.amax(block_points, axis=0, keepdims=True)
        #block_center = (block_min + block_max) / 2
        #block_center[0][-1] = block_min[0][-1]
        #block_points = block_points - block_center  # align to block bottom center
        x, y, z = np.split(block_points, (1, 2), axis=-1)

        block_xzyrgbi = np.concatenate([x, z, y, i[point_indices]], axis=-1)

        for block_split_idx in range(block_split_num):
            start = starts[block_split_idx]
            point_num = point_nums[block_split_idx]
            #print(block_split_num, block_split_idx, point_num )



            end = start + point_num
            idx_in_batch = idx % batch_size
            data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[start:end, :]
            data_num[idx_in_batch] = point_num
            indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]

            #print("indices_split_to_full", idx_in_batch, point_num, indices_split_to_full)

            if  (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1): #Last iteration

                item_num = idx_in_batch + 1
                
            idx = idx + 1
            
    return label_length, data, data_num, indices_split_to_full, item_num, all_label_pred, indices_for_prediction

def main(FLAG):
    
    data_filename = FLAG.filename
    ground_removed = FLAG.ground_removed
    retrieve_whole_files = FLAG.retrieve_whole_files
    
    
    if(FLAGS.postfix == "denoise-weights"):
        CHECKPOINT_LOCATION = "/home/hasan/data/hdd8TB/paper4-bosch-annotation/smart-annotation/pointcnn-models/kitti-only/denoise/0.99273694-iter--246000"
    else: # (FLAGS.postfix == "normal-weights"):
        CHECKPOINT_LOCATION = "/home/hasan/data/hdd8TB/paper4-bosch-annotation/smart-annotation/pointcnn-models/kitti-only/normal/0.98931193-iter--143000"
    
    max_point_num = 8192

    
    
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
    args_model = "pointcnn_seg"
    args_setting = "kitti3d_x8_2048_fps"
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

            label_length, data, data_num, indices_split_to_full, item_num, all_label_pred, indices_for_prediction = data_preprocessing(drivename, fname , max_point_num, ground_removed, FLAG)


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
            
            
            #bounded_indices =   (all_label_pred == 2 ).nonzero()[0]

            
            bounded_indices =   (all_label_pred > 0 ).nonzero()[0]


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
        default="normal-weights",
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