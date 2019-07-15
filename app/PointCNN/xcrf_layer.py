"""
MIT License

Copyright (c) 2019 Hasan Asyari Arief

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math as m


def _compatibility_initializer(shape):
    return  np.eye(shape[0], shape[1], dtype=np.float32)  *-1


class learningBlock():
    """ Implements the XCRF layer described in:


    """

    def __init__(self, num_points, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,point_indices, K=64, **kwargs):

        self.K_NN = K
        self.point_indices = point_indices
        self.num_points = num_points
        self.num_classes = num_classes
        
        with tf.variable_scope("crf_ker_weights", reuse=None):
            theta_alpha = tf.Variable( theta_alpha   , name="theta_alpha", trainable=False) 
            theta_beta = tf.Variable( theta_beta   , name="theta_beta", trainable=False) 
            theta_gamma = tf.Variable( theta_gamma   , name="theta_gamma", trainable=False) 

        
        self.theta_alpha = tf.square(theta_alpha) * 2 
        self.theta_beta = tf.square(theta_beta) * 2 
        self.theta_gamma = tf.square(theta_gamma) * 2
        
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None

        with tf.variable_scope("crf_ker_weights", reuse=None):
            self.compatibility_matrix_reset = tf.Variable( (_compatibility_initializer([num_classes,num_classes]) +1 ) , name="compatibility_matrix_reset", trainable=False) 

    def call(self, unaries, points_augmented, features_augmented, numberOfDimension, D=1):

        K_NN = self.K_NN

        tag = "fspace_"

        # Atrous Indices
        indices = self.point_indices[:,:,:K_NN*D,:]
        indices = indices[:,:,::D,:]


        #features_augmented  = tf.nn.softmax(features_augmented, -1)
        
        nn_f =  tf.gather_nd(features_augmented, indices, name=tag + 'nn_f')  # (N, P, K, 3)
        nn_f_center = tf.expand_dims(features_augmented, axis=2, name=tag + 'nn_f_center')  # (N, P, 1, 3)
            
        nn_pts = tf.gather_nd(points_augmented, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
        nn_pts_center = tf.expand_dims(points_augmented, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)


        #LOCAL XYZ Features
        nn_pts_local_xyz = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

        d_pts = ( tf.reduce_sum( tf.square(( nn_pts_local_xyz ) ), axis=-1) )  # | Pi - Pj |2 # (N, P, K)

        nn_f_local = tf.subtract(nn_f, nn_f_center, name=tag + 'nn_f_local')  # (N, P, K, 3)

        d_f = ( tf.reduce_sum( tf.square( ( nn_f_local ) ), axis=-1) ) # | Ii - Ij |2  # (N, P, K)


        q_values = unaries

        with tf.variable_scope("crf_ker_weights", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("appearance_k", reuse=tf.AUTO_REUSE):

                appearance_k = tf.get_variable("weights", shape=[ self.num_classes],
                   initializer=tf.initializers.ones(), trainable=True)


            with tf.variable_scope("smoothness_k", reuse=tf.AUTO_REUSE):

                smoothness_k = tf.get_variable("weights", shape=[self.num_classes],
                   initializer=tf.initializers.ones(), trainable=True)

            with tf.variable_scope("compatibility_kernel", reuse=tf.AUTO_REUSE):

                compatibility_kernel = tf.get_variable("weights", 
                   initializer=tf.Variable(initial_value = _compatibility_initializer((self.num_classes, self.num_classes)) ), 
                   dtype=tf.float32, trainable=True)


        compatibility_kernel = compatibility_kernel * self.compatibility_matrix_reset 

        for i in range(self.num_iterations):

            softmax_unaries = tf.nn.softmax(q_values) # (N, P, C)
            
            nn_softmax_unaries = tf.gather_nd(softmax_unaries, indices, name=tag + 'unaries')  # (N, P, K, C)
            
             
            predictions = tf.argmax(nn_softmax_unaries, axis=-1) # (N, P, K)
            compatibility_control = tf.one_hot( predictions, self.num_classes) # (N, P, K, C)
            
            compatibility_control = tf.reshape(compatibility_control, [-1,  self.num_classes])
            
            compatibility_control = tf.matmul(compatibility_control, compatibility_kernel)
            
            compatibility_control = tf.reshape(compatibility_control, [-1, self.num_points, K_NN, self.num_classes])
            


            bilateral_out =   tf.math.exp( -1 * (  tf.truediv(d_pts, self.theta_alpha) ) - ( tf.truediv(d_f, self.theta_beta ) ) )  # B, P, K

            bilateral_out = tf.expand_dims(bilateral_out, axis=-1)
            bilateral_out = tf.concat( [ bilateral_out for _ in range(self.num_classes) ], axis=-1)


            spatial_out =   tf.math.exp( -1 * tf.truediv(d_pts, self.theta_gamma) ) # B, P, K
            spatial_out = tf.expand_dims(spatial_out, axis=-1)
            spatial_out = tf.concat( [ spatial_out for _ in range(self.num_classes) ], axis=-1)


            # Message Passing
            k_fi_fj = bilateral_out * appearance_k + spatial_out * smoothness_k # B, P, K, C

            pairwise = tf.reduce_sum( k_fi_fj * nn_softmax_unaries * compatibility_control, axis=-2) #- softmax_unaries # B, P, C

            # Adding unary potentials
            q_values = unaries - pairwise

        return q_values