from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
import tensorflow as tf
from pointcnn import PointCNN

class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        
        
        print("points", points.get_shape())
        PointCNN.__init__(self, points, features, is_training, setting)
        
        per_loc_bin_num = int(setting.LOC_SCOPE / setting.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(setting.LOC_Y_SCOPE / setting.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + setting.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not setting.LOC_Y_BY_BIN else loc_y_bin_num * 2)
        
        
        channel_in = ( self.layer_fts[-1] if not setting.IS_FC_INPUT else  self.fc_layers[-1] )
        
        
        #Normalizing dimension for 1D
        channel_in = tf.reduce_mean(channel_in, axis=1,keep_dims=True,  name='fc_mean')
        
        
        print("channel_in", channel_in.shape)
        pre_channel = channel_in
        REG_FC = [256, 256]
        for k in range(0, len(setting.REG_FC)):
            
            pre_channel = pf.conv1d(pre_channel, setting.REG_FC[k], "CLS_LAYERS_"+str(k), is_training, with_bn=True)
            
        #pre_channel = pf.dense(pre_channel, reg_channel, "CLS_LAYERS_FC", is_training, with_bn=False, activation=None)
        
        pre_channel = pf.conv1d(pre_channel, reg_channel, "CLS_LAYERS_FC", is_training,  activation=None)
        
        if setting.DP_RATIO >= 0:
            pre_channel = tf.layers.dropout(pre_channel, setting.DP_RATIO, training=is_training, name='fc_reg_drop')
            
        #print("pre_channel", reg_channel, pre_channel.get_shape())
        self.logits = tf.transpose( pre_channel, perm=(0, 2, 1) ) 
        self.logits = tf.squeeze(self.logits,axis=-1)
        
        
        #print("logits", self.logits.get_shape())
        
        
        