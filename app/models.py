#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from sklearn.metrics import pairwise_distances

def range_overlap(
    a_min,
    a_max,
    b_min,
    b_max,
    ):
    '''Neither range is completely greater than the other
....'''

    return a_min <= b_max and b_min <= a_max


# https://codereview.stackexchange.com/questions/31352/overlapping-rectangles

def is_overlap(corners_new, corners):
    '''Overlapping rectangles overlap both horizontally & vertically
....'''

    r1 = {
        'left': np.min(corners[:, 0]),
        'right': np.max(corners[:, 0]),
        'bottom': np.min(corners[:, 1]),
        'top': np.max(corners[:, 1]),
        }
    r2 = {
        'left': np.min(corners_new[:, 0]),
        'right': np.max(corners_new[:, 0]),
        'bottom': np.min(corners_new[:, 1]),
        'top': np.max(corners_new[:, 1]),
        }

    return range_overlap(r1['left'], r1['right'], r2['left'], r2['right'
                         ]) and range_overlap(r1['bottom'], r1['top'],
            r2['bottom'], r2['top'])


def is_overlap_with_other_boxes(box_id, corner_checks, other_boxes):

    for bbox in other_boxes:
        if box_id != bbox.id:
            corners = bbox.get_corners()
            if is_overlap(corners, corner_checks):
                return True
    return False


def fixed_annotation_error(json_data, with_kalman_filter=True):

    json_bounding_boxes = json_data['frame']['bounding_boxes']

    #print ('prior', json_bounding_boxes)
    frame = Frame.parse_json(json_data)

    H = frame.H_MATRIX
    R = frame.R_MATRIX
    F = frame.F_MATRIX
    Q = frame.Q_MATRIX
    
    
        
    #print ('F', F)
    #print ('Q', Q)
    #print ('R', R)

    dt = frame.dt

    box_id = -1
    for bounding_box in frame.bounding_boxes:
        box_id = box_id + 1

        # Previous state initialization

        z_k = bounding_box.center
        x_k_min_1 = bounding_box.predicted_state
        P_k_min_1 = bounding_box.predicted_error

        
        if(np.sum(x_k_min_1[:2])== 0):
            x_k_min_1[:2] = z_k
            x_k_min_1[2:] = 0
            
        """    
        x_k_min_1[2] = (x_k_min_1[0] - z_k[0]) / dt  #V_x 
        x_k_min_1[3] = (x_k_min_1[1] - z_k[1]) / dt  #V_y 
        x_k_min_1[4] = (x_k_min_1[2] - bounding_box.predicted_state[2]) / dt  #A_x 
        x_k_min_1[5] = (x_k_min_1[3] - bounding_box.predicted_state[3]) / dt  #A_y 
        """
        #print ('predicted_state', bounding_box.predicted_state)
        #print ('predicted_error', bounding_box.predicted_error)
        print ('x_k_min_1', x_k_min_1, x_k_min_1.shape)
        
        x_hat_k_prior = np.matmul(F, x_k_min_1)
        
        P_k_prior = np.matmul(np.matmul(F, P_k_min_1), np.transpose(F)) \
            + Q

        
        #print ('x_hat_k_prior', x_hat_k_prior)
        #print ('P_k_prior', P_k_prior)
        
        y_k = z_k - np.matmul(H, x_hat_k_prior)

        
        print ('z_k', z_k)
        print ('y_k', y_k)
        
        _temp = np.linalg.inv(R + np.matmul(np.matmul(H, P_k_prior),
                              np.transpose(H)))
        K_k = np.matmul(np.matmul(P_k_prior, np.transpose(H)), _temp)

        x_hat_k = x_hat_k_prior + np.matmul(K_k, y_k)

        
        #Force using previous velocity and acceleration
        
        v_x = (x_hat_k[0] - x_k_min_1[0]) / dt
        v_y = (x_hat_k[1] - x_k_min_1[1]) / dt
        
        
        a_x = (v_x - x_k_min_1[2]) / dt
        a_y = (v_y - x_k_min_1[3]) / dt
        
        x_hat_k[2:] = [v_x, v_y, a_x, a_y]
        
        #print(x_hat_k)
        _temp = np.eye(6) - np.matmul(K_k, H)
        P_k = np.matmul(np.matmul(_temp, P_k_prior),
                        np.transpose(_temp)) + np.matmul(np.matmul(K_k,
                R), np.transpose(K_k))

        #print ('P_k', P_k)
        #print ('K_k', K_k)
        print ('x_hat_k', x_hat_k)
        
        json_bounding_boxes[box_id]['center']['x'] = x_hat_k[0]
        json_bounding_boxes[box_id]['center']['y'] = x_hat_k[1]
        
        json_bounding_boxes[box_id]['predicted_error'] = \
            np.diag(P_k).tolist()
        json_bounding_boxes[box_id]['predicted_state'] = \
            x_hat_k.tolist()

    json_data['frame']['bounding_boxes'] = json_bounding_boxes
    #print ('updated-kalman', json_bounding_boxes)
    return json_data


class NextFrameBBOX:

    def __init__(
        self,
        box_id,
        back_tracking_boxes,
        box_state,
        center_location,
        tracking_idx,
        is_bbox_updated=False,
        ):
        self.id = box_id
        self.tracking_idx = tracking_idx
        self.box_id = box_id
        self.is_bbox_updated = is_bbox_updated
        self.center = center_location
        self.back_tracking_boxes = back_tracking_boxes
        self.box_state = box_state
        self.box_track_indices = sorted(back_tracking_boxes.keys())
        self.current_box_track_index = len(self.box_track_indices) - 1  # from very last

    def update_index(self):
        self.current_box_track_index = self.current_box_track_index - 1
        if self.current_box_track_index < 0:
            self.current_box_track_index = 0
        self.is_bbox_updated = True
        return self.current_box_track_index

    def get_tracking_index(self):
        return self.box_track_indices[self.current_box_track_index]

    def get_corners(self):
        return self.back_tracking_boxes[self.get_tracking_index()][0]

    def get_center_dist(self):
        return self.back_tracking_boxes[self.get_tracking_index()][1]

    def get_bounding_box(self, bbox):

        bbox['center_dist'] = self.get_center_dist()
        bbox['object_id'] = self.box_state['object_id']
        bbox['predicted_state'] = self.box_state['predicted_state']
        bbox['predicted_error'] = self.box_state['predicted_error']
        bbox['tracking_idx'] = self.box_state['tracking_idx']
        return bbox

    def is_boxes_overlap(self, box_check):
        return is_overlap(self.get_corners(), box_check.get_corners())


class Frame:

    def __init__(
        self,
        fname,
        bounding_boxes,
        dt=0.1,
        ):
        self.fname = fname
        self.bounding_boxes = bounding_boxes
        self.dt =dt
        self.F_MATRIX = np.array([[ 1, 0, dt, 0, 0.5 * dt * dt, 0, ], 
                                  [ 0, 1, 0, dt, 0, 0.5 * dt * dt, ], 
                                  [ 0, 0, 1, 0, dt, 0, ], 
                                  [ 0, 0, 0, 1, 0, dt, ], 
                                  [ 0, 0, 0, 0, 1, 0, ], 
                                  [ 0, 0, 0, 0, 0, 1, ], ], dtype=np.float32)
        self.Q_MATRIX = np.eye(6) * [ 0, 0, 0, 0, 0, 0, ]
        self.R_MATRIX = np.eye(2) * [0.0000000001, 0.0000000001]
        self.H_MATRIX = np.array([[ 1, 0, 0, 0, 0, 0, ], 
                                  [ 0, 1, 0, 0, 0, 0, ]], dtype=np.float32)

    @staticmethod
    def parse_json(json_frame):
        json_bounding_boxes = json_frame['frame']['bounding_boxes']
        bounding_boxes = BoundingBox.parse_json(json_bounding_boxes)
        return Frame(json_frame['frame']['fname'], bounding_boxes)


class BoundingBox:

    def __init__(
        self,
        box_id,
        center,
        height,
        width,
        length,
        angle,
        object_id,
        predicted_state,
        predicted_error,
        settingsControls,
        tracking_idx,
        timestamps,
        islocked
        ):
        self.box_id = box_id
        self.x = center['x']
        self.y = center['y']
        self.center = np.array([self.x, self.y])
        self.height = height
        self.width = width
        self.length = length
        self.angle = angle
        self.settingsControls = settingsControls
        self.object_id = object_id
        self.predicted_error = np.eye(6) * np.array(predicted_error, dtype=np.float32)
        self.predicted_state = np.transpose(np.array(predicted_state, dtype=np.float32))
        self.tracking_idx = tracking_idx
        self.islocked = islocked
        self.timestamps = timestamps

    @staticmethod
    def parse_json(json):
        return [BoundingBox(
            json_obj['box_id'],
            json_obj['center'],
            json_obj['height'],
            json_obj['width'],
            json_obj['length'],
            json_obj['angle'],
            json_obj['object_id'],
            json_obj['predicted_state'],
            json_obj['predicted_error'],
            json_obj['settingsControls'],
            json_obj['tracking_idx'],
            json_obj['timestamps'],
            json_obj['islocked'],
            ) for json_obj in json]

    def filter_points(self, pointcloud, bounding_factor=.1):
        (l, w, theta) = (self.length, self.width, self.angle)
        center = np.array([[self.x, self.y]])
        rotated_points = pointcloud.rigid_transform(theta, center)
        (x, y) = (rotated_points[:, 0], rotated_points[:, 1])
        indices_within_width = np.where(np.abs(x) <= w / 2 * (1
                + bounding_factor))[0]
        indices_within_length = np.where(np.abs(y) <= l / 2 * (1
                + bounding_factor))[0]

        bounded_indices = np.intersect1d(indices_within_width,
                indices_within_length)
        return bounded_indices

    def grow_pointcloud(self, pointcloud):
        
        _, filtered_pc = self.filter_pointcloud(np.copy(pointcloud))
        D = pairwise_distances(pointcloud[:,:2], filtered_pc[:,:2])
        _min_results = np.amin(D, axis=1)
        print("D", D.shape, _min_results.shape)
    
        
        
    def filter_pointcloud(self, pointcloud, updated_size = 0):
        theta = self.angle
        transformed_pointcloud = homogeneous_transformation(pointcloud,
                self.center, -theta)
        indices = \
            np.intersect1d(np.where(np.abs(transformed_pointcloud[:,
                           0]) <= (self.width+updated_size) / 2)[0],
                           np.where(np.abs(transformed_pointcloud[:,
                           1]) <= (self.length+updated_size) / 2)[0])
        return (np.delete(pointcloud, indices, axis=0),
                pointcloud[indices, :])

    def get_corners(self):
        c1 = np.array([-self.width / 2, -self.length / 2])
        c2 = np.array([self.width / 2, -self.length / 2])
        c3 = np.array([self.width / 2, self.length / 2])
        c4 = np.array([-self.width / 2, self.length / 2])
        corners = homogeneous_transformation(np.vstack([c1, c2, c3,
                c4]), np.zeros(2), self.angle) + self.center
        return corners


def homogeneous_transformation(points, translation, theta):
    return (points[:, :2] - translation).dot(rotation_matrix(theta).T)


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),
                    np.cos(theta)]])


