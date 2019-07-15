#!/usr/bin/python3
"""This file is modified from PointCNN training segmentation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

from xcrf_layer import learningBlock

import data_utils
import pointfly as pf


import confusion_matrix
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to training set ground truth (.txt)', required=True)
    parser.add_argument('--filelist_val', '-v', help='Path to validation set ground truth (.txt)', required=True)
    parser.add_argument('--filelist_unseen', '-u', help='Path to validation set ground truth (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%s_%d' % (args.model, args.setting, time_string, os.getpid()))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    print('PID:', os.getpid())

    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    sample_num = setting.sample_num
    step_val = setting.step_val
    label_weights_list = setting.label_weights
    label_weights_val = [1.0] * 1 + [1.0] * (setting.num_class - 1) 

    rotation_range = setting.rotation_range
    rotation_range_val = setting.rotation_range_val
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    is_list_of_h5_list = not data_utils.is_h5_list(args.filelist)
    if is_list_of_h5_list:
        seg_list = data_utils.load_seg_list(args.filelist)
        seg_list_idx = 0
        filelist_train = seg_list[seg_list_idx]
        seg_list_idx = seg_list_idx + 1
    else:
        filelist_train = args.filelist
    data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
    data_val, _, data_num_val, label_val, _ = data_utils.load_seg(args.filelist_val)

    data_unseen, _, data_num_unseen, label_unseen, _ = data_utils.load_seg(args.filelist_unseen)

    # shuffle
    data_unseen, data_num_unseen, label_unseen = \
        data_utils.grouped_shuffle([data_unseen, data_num_unseen, label_unseen])


    # shuffle
    data_train, data_num_train, label_train = \
        data_utils.grouped_shuffle([data_train, data_num_train, label_train])

    num_train = data_train.shape[0]
    point_num = data_train.shape[1]
    num_val = data_val.shape[0]
    num_unseen = data_unseen.shape[0]

    batch_num_unseen = int(math.ceil(num_unseen / batch_size) )



    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))
    batch_num = (num_train * num_epochs + batch_size - 1) // batch_size
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
    batch_num_val = int(math.ceil(num_val / batch_size) )
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    is_labelled_data = tf.placeholder(tf.bool, name='is_labelled_data')


    pts_fts = tf.placeholder(tf.float32, shape=(None, point_num, setting.data_dim), name='pts_fts')
    labels_seg = tf.placeholder(tf.int64, shape=(None, point_num), name='labels_seg')
    labels_weights = tf.placeholder(tf.float32, shape=(None, point_num), name='labels_weights')

    ######################################################################
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

    labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
    labels_weights_sampled = tf.gather_nd(labels_weights, indices=indices, name='labels_weight_sampled')

    net = model.Net(points_augmented, features_augmented, is_training, setting)

    logits = net.logits
    probs = tf.nn.softmax(logits, name='prob')
    predictions = tf.argmax(probs, axis=-1, name='predictions')

    with tf.variable_scope('xcrf_ker_weights'):
        
        _, point_indices = pf.knn_indices_general(points_augmented, points_augmented, 1024, True)

        
        xcrf = learningBlock(num_points=sample_num,
                         num_classes=setting.num_class,
                         theta_alpha=float(5),
                         theta_beta=float(2),
                         theta_gamma=float(1),
                         num_iterations=5,
                         name='xcrf',
                         point_indices=point_indices
                         )
        
        _logits1 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim)
        _logits2 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim, D=2)
        _logits3 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim, D=3)
        _logits4 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim, D=4)
        _logits5 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim, D=8)
        _logits6 = xcrf.call(net.logits, points_augmented, features_augmented, setting.data_dim, D=16)
    
        _logits = _logits1 + _logits2 + _logits3+ _logits4 + _logits5 + _logits6
        _probs = tf.nn.softmax(_logits, name='probs_crf')
        _predictions = tf.argmax(_probs, axis=-1, name='predictions_crf')
                
    
    logits = tf.cond(is_training, 
    	lambda:tf.cond(is_labelled_data, lambda:_logits, lambda:logits), 
    	lambda:_logits) 

   
    predictions = tf.cond(is_training, 
    	lambda:tf.cond(is_labelled_data, lambda:_predictions, lambda:predictions), 
    	lambda:_predictions) 
    
    labels_sampled = tf.cond(is_training, 
    	lambda:tf.cond(is_labelled_data, lambda:labels_sampled, lambda:_predictions), 
    	lambda:labels_sampled) 



    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_sampled, logits=logits,
                                                     weights=labels_weights_sampled)

    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_sampled, predictions, weights=labels_weights_sampled)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = \
            tf.metrics.mean_per_class_accuracy(labels_sampled, predictions, setting.num_class,
                                               weights=labels_weights_sampled)
        t_1_per_mean_iou_op, t_1_per_mean_iou_op_update_op = \
            tf.metrics.mean_iou(labels_sampled, predictions, setting.num_class,
                                               weights=labels_weights_sampled)
        
    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])


    _ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
    _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    _ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
    _ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])
    _ = tf.summary.scalar('t_1_mean_iou/val', tensor=t_1_per_mean_iou_op, collections=['val'])


    all_variable = tf.global_variables()

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)



    xcrf_ker_weights =  [var for var in tf.global_variables() if 'xcrf_ker_weights' in var.name] 
    no_xcrf_ker_weights =  [var for var in tf.global_variables() if 'xcrf_ker_weights' not in var.name]  
    
    #print(restore_values)
        #train_op = optimizer.minimize( loss_op+reg_loss, global_step=global_step)
        
    
    with tf.control_dependencies(update_ops):
        train_op_xcrf = optimizer.minimize(loss_op+ reg_loss, var_list=no_xcrf_ker_weights, global_step=global_step)
        train_op_all = optimizer.minimize(loss_op+ reg_loss, var_list=all_variable, global_step=global_step)

    train_op = tf.cond(is_labelled_data, lambda:train_op_all, lambda:train_op_xcrf)


    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(var_list=no_xcrf_ker_weights, max_to_keep=5)

    saver_best = tf.train.Saver(max_to_keep=5)
    # backup all code
    code_folder = os.path.abspath(os.path.dirname(__file__))
    shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_ckpt_best = os.path.join(root_folder, 'ckpt-best')
    if not os.path.exists(folder_ckpt_best):
        os.makedirs(folder_ckpt_best)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), int(parameter_num)))


    _highest_val = 0.0
    max_val = 10
    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_values_op = tf.summary.merge_all('summary_values')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        batch_num = 50000 
        for batch_idx_train in range(batch_num):
            if (batch_idx_train % step_val == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) \
                    or batch_idx_train == batch_num - 1:
                ######################################################################
                
                # Validation
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                sess.run(reset_metrics_op)
                summary_hist = None
                _idxVal = np.arange(num_val)
                np.random.shuffle(_idxVal)

                _pred = []
                _label = []
                for batch_val_idx in tqdm(range(batch_num_val)):
                    start_idx = batch_size * batch_val_idx
                    end_idx = min(start_idx + batch_size, num_val)
                    batch_size_val = end_idx - start_idx
                    
                    points_batch = data_val[ _idxVal[start_idx:end_idx], ...]
                    points_num_batch = data_num_val[ _idxVal[start_idx:end_idx], ...]
                    labels_batch = label_val[ _idxVal[start_idx:end_idx], ...]

                    weights_batch = np.array(label_weights_val)[labels_batch]
                    
                    
                    xforms_np, rotations_np = pf.get_xforms(batch_size_val,
                                                            rotation_range=rotation_range_val,
                                                            scaling_range=scaling_range_val,
                                                            order=setting.rotation_order)


                    _labels_sampled, _predictions, _, _, _, _ = sess.run([labels_sampled, predictions, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op, t_1_per_mean_iou_op_update_op],
                             feed_dict={
                                 pts_fts: points_batch,
                                 indices: pf.get_indices(batch_size_val, sample_num, points_num_batch),
                                 xforms: xforms_np,
                                 rotations: rotations_np,
                                 jitter_range: np.array([jitter_val]),
                                 labels_seg: labels_batch,
                                 is_labelled_data: True,
                                 labels_weights: weights_batch,
                                 is_training: False,
                             })
                    _pred.append(_predictions.flatten())
                    _label.append(_labels_sampled.flatten())


                loss_val, t_1_acc_val, t_1_per_class_acc_val, t1__mean_iou, summaries_val = sess.run(
                    [ loss_mean_op, t_1_acc_op, t_1_per_class_acc_op, t_1_per_mean_iou_op, summaries_val_op])

                img_d_summary = pf.plot_confusion_matrix(_label, _pred, ["Environment", "Pedestrian", "Car", "Cyclist"], tensor_name='confusion_matrix')
                
                max_val = max_val -1

                if( t_1_per_class_acc_val > _highest_val ):

                    max_val = 10

                    _highest_val = t_1_per_class_acc_val

                    filename_ckpt = os.path.join(folder_ckpt_best, str(_highest_val)+"-iter-")
                    saver_best.save(sess, filename_ckpt, global_step=global_step)


                if(max_val < 0):
                    sys.exit(0)



                summary_writer.add_summary(summaries_val, batch_idx_train)

                summary_writer.add_summary(img_d_summary, batch_idx_train)

                #summary_writer.add_summary(summary_hist, batch_idx_train)
                print('{}-[Val  ]-Average:      Loss: {:.4f}  T-1 Acc: {:.4f}  Diff-Best: {:.4f} T-1 mAcc: {:.4f}   T-1 mIoU: {:.4f}'
                      .format(datetime.now(), loss_val, t_1_acc_val, _highest_val-t_1_per_class_acc_val, t_1_per_class_acc_val, t1__mean_iou))
                sys.stdout.flush()
                ######################################################################

                
                # Unseen Data
                sess.run(reset_metrics_op)
                summary_hist = None
                _idxunseen = np.arange(num_unseen)
                np.random.shuffle(_idxunseen)

                _pred = []
                _label = []
                
                unseenIndices = np.arange(batch_num_unseen)
                np.random.shuffle(unseenIndices)
                for batch_val_idx in tqdm(unseenIndices[:500]):
                    start_idx = batch_size * batch_val_idx
                    end_idx = min(start_idx + batch_size, num_unseen)
                    batch_size_train = end_idx - start_idx
                    
                    points_batch = data_unseen[ _idxunseen[start_idx:end_idx], ...]
                    points_num_batch = data_num_unseen[ _idxunseen[start_idx:end_idx], ...]
                    labels_batch = np.zeros(points_batch.shape[0:2], dtype=np.int32)

                    weights_batch = np.array(label_weights_list)[labels_batch]

                    xforms_np, rotations_np = pf.get_xforms(batch_size_train,
                                                            rotation_range=rotation_range,
                                                            scaling_range=scaling_range,
                                                            order=setting.rotation_order)

                    offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
                    offset = max(offset, -sample_num * setting.sample_num_clip)
                    offset = min(offset, sample_num * setting.sample_num_clip)
                    sample_num_train = sample_num + offset

                    sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op, t_1_per_mean_iou_op_update_op],
		                     feed_dict={
		                         pts_fts: points_batch,
		                         is_labelled_data:False,
		                         indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
		                         xforms: xforms_np,
		                         rotations: rotations_np,
		                         jitter_range: np.array([jitter]),
		                         labels_seg: labels_batch,
		                         labels_weights: weights_batch,
		                         is_training: True,
		                     })
                    if batch_val_idx % 100 == 0:
		                loss, t_1_acc, t_1_per_class_acc, t_1__mean_iou = sess.run([loss_mean_op,
		                                                                        t_1_acc_op,
		                                                                        t_1_per_class_acc_op,
		                                                                        t_1_per_mean_iou_op],
		                     feed_dict={
		                         pts_fts: points_batch,
		                         indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
		                         xforms: xforms_np,
		                         is_labelled_data:False,
		                         rotations: rotations_np,
		                         jitter_range: np.array([jitter]),
		                         labels_seg: labels_batch,
		                         labels_weights: weights_batch,
		                         is_training: True,
		                     })
		                print('{}-[Train]-Unseen: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}   T-1 mIoU: {:.4f}'
		                      .format(datetime.now(), batch_val_idx, loss, t_1_acc, t_1_per_class_acc, t_1__mean_iou))
	                	sys.stdout.flush()



            

            ######################################################################
            # Training
            start_idx = (batch_size * batch_idx_train) % num_train
            end_idx = min(start_idx + batch_size, num_train)
            batch_size_train = end_idx - start_idx
            points_batch = data_train[start_idx:end_idx, ...]
            points_num_batch = data_num_train[start_idx:end_idx, ...]
            labels_batch = label_train[start_idx:end_idx, ...]
            weights_batch = np.array(label_weights_list)[labels_batch]

            if start_idx + batch_size_train == num_train:
                if is_list_of_h5_list:
                    filelist_train_prev = seg_list[(seg_list_idx - 1) % len(seg_list)]
                    filelist_train = seg_list[seg_list_idx % len(seg_list)]
                    if filelist_train != filelist_train_prev:
                        data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
                        num_train = data_train.shape[0]
                    seg_list_idx = seg_list_idx + 1
                data_train, data_num_train, label_train = \
                    data_utils.grouped_shuffle([data_train, data_num_train, label_train])

            offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
            offset = max(offset, -sample_num * setting.sample_num_clip)
            offset = min(offset, sample_num * setting.sample_num_clip)
            sample_num_train = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train,
                                                    rotation_range=rotation_range,
                                                    scaling_range=scaling_range,
                                                    order=setting.rotation_order)
            sess.run(reset_metrics_op)
            sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op, t_1_per_mean_iou_op_update_op],
                     feed_dict={
                         pts_fts: points_batch,
                         is_labelled_data:True,
                         indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
                         xforms: xforms_np,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter]),
                         labels_seg: labels_batch,
                         labels_weights: weights_batch,
                         is_training: True,
                     })
            if batch_idx_train % 100 == 0:
                loss, t_1_acc, t_1_per_class_acc, t_1__mean_iou, summaries = sess.run([loss_mean_op,
                                                                        t_1_acc_op,
                                                                        t_1_per_class_acc_op,
                                                                        t_1_per_mean_iou_op,
                                                                        summaries_op],
                     feed_dict={
                         pts_fts: points_batch,
                         indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
                         xforms: xforms_np,
                         is_labelled_data:True,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter]),
                         labels_seg: labels_batch,
                         labels_weights: weights_batch,
                         is_training: True,
                     })
                summary_writer.add_summary(summaries, batch_idx_train)
                print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}   T-1 mIoU: {:.4f}'
                      .format(datetime.now(), batch_idx_train, loss, t_1_acc, t_1_per_class_acc, t_1__mean_iou))
                sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()