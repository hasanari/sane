from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
from pointcnn import PointCNN
import tensorflow as tf

class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        self.logits = pf.dense(self.fc_layers[-1], setting.num_class, 'logits',
                               is_training, with_bn=False, activation=None)
        
        #self.feature_logits = pf.dense(self.fc_layers[0], setting.num_class, 'feature_logits',
        #                       is_training, with_bn=False, activation=None)
        
        #self.logits  = self.logits  * tf.nn.softmax(self.feature_logits , name='probs_feature_logits')