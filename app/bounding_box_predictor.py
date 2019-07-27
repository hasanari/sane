import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial import cKDTree
from models import BoundingBox, Frame
from os.path import join, isfile
from os import listdir
from oxt import load_oxts_lite_data, oxts2pose
from frame_handler import FrameHandler
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import math

from mask_rcnn import get_pointcnn_labels

CAR_SHAPE = {
   # "suv" : {"width": 4.398504570721579, "length": 1.7581043589275447},
    "suv" : {"width": 3.45, "length": 1.77},
    "truck" : {},
    "truck" : {},
    "truck" : {}
}




# This Parameter only for Car objects 
"""
distribution_pts = [     2.1770e+03,   2.1770e+03,  2177.        , 2840.01764706,
       2591.44791667, 2620.98920863, 1691.25454545, 1156.63664596,
        797.61849711,  666.90149626,  612.04436451,  524.61043689,
        466.82988506,  377.10612691,  315.22765073,  261.31118143,
        222.88185654,  197.18070953,  167.83592018,  153.9748996 ,
        133.69254658,  111.69578313,   99.31440162,   86.47524752,
         81.00598802,   72.21800434,   62.41469194,   58.53493976,
         52.56440281,   48.3381295 ,   46.08064516,   39.30787589,
         37.32566586,   33.92772512,   31.80933333,   30.35775862,
         27.99710983,   25.87605634,   24.95076923,   23.19      ,
         21.95032051,   20.1462585 ,   20.01737452,   19.75830258,
         16.38157895,   15.82806324,   14.97560976,   14.01746725,
         12.71527778,   12.3480663 ,   12.33116883,   11.57803468,
         11.11949686,   10.20486111,   10.03741497,    8.72836538,
          8.84586466,    9.05701754,    8.33139535,    8.36238532,
          8.29166667,    8.32954545,    7.63186813,    8.0625    ,
          7.7734375 ,    7.44067797,    6.69166667,    6.64754098,
          6.76923077,    7.175     ,    6.84313725,    6.90789474,
          6.56382979,    6.7625    ,    6.29411765,    6.19230769,
          6.33333333,    6.91666667,    6.03333333,    7.        ,
          5.95454545,    5.8       ,    5.        ]
"""

from sklearn.metrics import pairwise_distances
IGNORE_NEIGBOR = 9999999999
UNCLASSIFIED = False
NOISE = None     
distribution_pts = [ 2177, 2177, 2177., 817., 788., 996., 519., 336., 109., 105., 121., 84., 93., 28., 15., 5., 5., 5., 5., 21., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]
max_distance = 4
volume = 22.0

def batch_distance_matrix(A, B):
    
    
    
    #A[:,:,2] = 0
    #B[:,:,2] = 0
    #A = A[:,:,:2]
    #B = B[:,:,:2]
    A = A[:,:,[0,2]]
    B = B[:,:,:[0,2]]
    r_A = np.sum(A * A, axis=2, keepdims=True)
    r_B = np.sum(B * B, axis=2, keepdims=True)
    m = np.matmul(A, np.transpose(B, axes=(0, 2, 1)))
    D = r_A - 2 * m + np.transpose(r_B, axes=(0, 2, 1))
    
    return D

def matrix_scan(m,  max_distance, min_points):


    m = np.concatenate( [ m[:,0:1], m[:,1:2] ], axis=-1 )
                            
    print("m", m.shape)
    _points = np.expand_dims(m, 0)
    #D = batch_distance_matrix(_points, _points)[0]
    
    D = pairwise_distances(m, m, 'manhattan')
    
    
    di = np.diag_indices(D.shape[0])
    D[di] = IGNORE_NEIGBOR # Remove all diagonal
    D[ D >= max_distance] = IGNORE_NEIGBOR # Remove all more than Maximum Distances between points
    #D[ D <= eps] = IGNORE_NEIGBOR # Remove all more than Minimum Distances between points
    
    
    #Include Volume Filtering
    
    considered_nn = D != IGNORE_NEIGBOR
    
    considered_nn_counts = [ len( considered_nn[point_id].nonzero()[0]) for point_id in range(D.shape[0])]
    

    input_counts = list(considered_nn_counts)
    #Include Origin Aware Minimum-PTS
    distance_origin = np.sqrt(m[:,0]*m[:,0] + m[:,1]*m[:,1]).astype(int)
    for _i in range(D.shape[0]):
        if( distance_origin[_i] < len(distribution_pts) ): #Only change the value within the length
            if( distribution_pts[ distance_origin[_i]  ] > considered_nn_counts[_i] ): #Check if min-pts acceptable
                considered_nn_counts[_i] = 0

    _opt = np.array(considered_nn_counts)
    

    clustered_list =  np.ones(_opt.shape[0]) - 2
    cluster_id = 0
    center_cluster = []
    while(max(_opt) > min_points):
        max_indices = np.argmax(_opt)
        found_object_idx = list(considered_nn[max_indices].nonzero()[0])
        found_object_idx.append(max_indices)

        center_cluster.append(max_indices)
        
        clustered_list[found_object_idx] = cluster_id

        _opt[found_object_idx] = 0
        cluster_id =  cluster_id +1


    return clustered_list.astype(int), center_cluster


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def newline(v1, v2, c='blue'):
    
    #p1 =[p1[1],p1[0]]
    #p2 =[p2[1],p2[0]]
    num_samples = 10

    x = np.linspace(v1[0], v2[0], num_samples)
    y = np.linspace(v1[1], v2[1], num_samples)
    plt.plot(y, x, c=c)
            
            
#https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
def rotate_origin_only(xy, radians):
    
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    
    #print("rotate_origin_only", xy, xx, yy, radians)
    
    return np.array([xx, yy])


def rotate_origin_only_bulk(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy[:,0], xy[:,1]
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return np.array([xx, yy]).transpose()


        
class BoundingBoxPredictor():
    def __init__(self, frame_handler):
        self.n_segs = (1,1)
        self.n_iter=5
        self.n_lpr=500
        self.th_seeds=.4
        self.th_dist=.2

        self.frame_handler = frame_handler
        
        self.oxt_path = "oxts/"
        self.oxts = {drive: load_oxts_lite_data(join(FrameHandler.DATASET_DIR, drive), self.frame_handler.drives[drive]) 
                    for drive in self.frame_handler.drives.keys()}
        self.poses = {drive: oxts2pose(self.oxts[drive], drive) for drive in tqdm(self.oxts.keys())}

    def transform_coords(self, fname, x, inv=False):
        if x.size == 2:
            x = np.append(x, [0, 1])
        if x.size == 3:
            x = np.append(x, [1])
        idx = self.frame_handler.frame_names.index(fname)
        transform = self.poses[idx]
        if inv:
            transform = np.linalg.inv(transform)

        return transform @ x

    def get_velocities(self, prev_frame, cur_frame, ref_fname):
        bounding_boxes = sorted(cur_frame.bounding_boxes, 
                                key=lambda box: box.box_id)
        velocities = {}
        prev_frame_bounding_boxes = {box.box_id:box for box in prev_frame.bounding_boxes}
        for i, box in enumerate(bounding_boxes):
            box_id = box.box_id
            #print(box_id)
            cur_center = box.center

            if box_id in prev_frame_bounding_boxes:
                prev_center = prev_frame_bounding_boxes[box_id].center
                cur_center_corr = self.transform_coords(cur_frame.fname, cur_center)
                prev_center_corr = self.transform_coords(prev_frame.fname, prev_center)
                velocities[box_id] = self.transform_coords(ref_fname, 
                                                           cur_center - prev_center,
                                                           inv=True)[:2]

        return velocities
    
    
    def fully_automated_bbox(self, fname, json_request):
        
        
        drivename, fname = fname.split("/")

        idx = self.frame_handler.drives[drivename].index(fname)       
        
        ground_removed  = False
    
        car_points = get_pointcnn_labels(drivename+"/"+fname, json_request["settingsControls"], ground_removed=ground_removed)

        pc = self.frame_handler.get_pointcloud(drivename, fname, dtype=float, ground_removed=ground_removed)
        
        points_class = pc[car_points]
        
        max_distance_per_class =1.0
        type_criterion =  1
        is_shape_fitting_required = False
        
        
        
        #object_ids, center_cluster = matrix_scan(points_class[ : ,:3], max_distance_per_class, 20)

        from sklearn.cluster import DBSCAN
        
        clustering = DBSCAN(eps=max_distance_per_class, min_samples=50, metric='euclidean').fit(points_class[ : ,:2])
        
        object_ids = clustering.labels_
        
        number_of_objects = max(object_ids)

        inc_obj = 0
        bbox_storage = []
        point_max_storage = []
        
        bounding_boxes_opt = {}
        for object_id in range(number_of_objects+1):
            inc_obj = inc_obj + 1 
            individual_object_indices  = object_ids == object_id


            png_source = points_class[individual_object_indices, :]

            centroid = [np.mean(png_source[:,0]), np.mean(png_source[:,1]), np.min(png_source[:,2])]


            X = png_source

            if( type_criterion == 0 ): 
                _criterion = area_criterion
            if( type_criterion == 1): 
                   _criterion = closeness_criterion
            elif(type_criterion == 3): 
                   _criterion = variance_criterion


            edges, corners = self.search_rectangle_fit(X, _criterion)


            bounding_box, pointsInside, corners = self.corners_to_bounding_box(corners, np.copy(png_source), is_shape_fitting_required)

            bounding_boxes_opt[str(object_id)] = bounding_box
                

        return bounding_boxes_opt
        
        
    def predict_next_frame_bounding_boxes(self, frame, json_request):
        drivename, fname = frame.fname.split('.')[0].split("/")

        idx = self.frame_handler.drives[drivename].index(fname)
        next_fname = self.frame_handler.drives[drivename][idx+1]

        
        
        ground_removed  = json_request["settingsControls"]["GroundRemoval"]
        
        #print("ground_removed", ground_removed)
    
        car_points = get_pointcnn_labels(drivename+"/"+fname, json_request["settingsControls"], ground_removed=ground_removed)
        
        pc = self.frame_handler.get_pointcloud(drivename, fname, dtype=float, ground_removed=ground_removed)
        
        
        
        pc = pc[car_points]
        
        
    
        car_points = get_pointcnn_labels(drivename+"/"+next_fname, json_request["settingsControls"], ground_removed=ground_removed)
        
        
        next_pc = self.frame_handler.get_pointcloud(drivename, next_fname, dtype=float, ground_removed=ground_removed)
        
        next_pc = next_pc[car_points]
        
        
        #print(fname, ground_removed)
        #print([box.box_id for box in frame.bounding_boxes])
        bounding_boxes = sorted(frame.bounding_boxes, 
                            key=lambda box:box.box_id)
        centers = {box.box_id:box.center for box in bounding_boxes}
        velocities = {box_id:np.zeros(2) for box_id in centers.keys()}
        
        next_pc[:,2] = 0
        next_pc = next_pc[:,:3]
        np.random.shuffle(next_pc)
        next_pc_small = next_pc[::4]
        next_bounding_boxes = {}
        for bounding_box in bounding_boxes:
            try:
                next_bounding_boxes[str(bounding_box.box_id)] = self._predict_next_frame_bounding_box(frame, bounding_box, next_pc_small) 
            except:
                pass

        # next_bounding_boxes = {str(bounding_box.box_id):self._predict_next_frame_bounding_box(bounding_box, next_pc_small) 
        #                         for bounding_box in bounding_boxes}
        return next_bounding_boxes

    def _predict_next_frame_bounding_box(self, frame, bounding_box, pc):
        start = time.time()
        
        """Pure state to state linear movement Kalman Filter"""
        z_k = bounding_box.center
        x_k =  bounding_box.predicted_state
        P_k =  bounding_box.predicted_error
      
        if(np.sum(x_k[:2])== 0):
            x_k[:2] = z_k
            
        H = frame.H_MATRIX
        R = frame.R_MATRIX

        y_k = z_k - np.matmul(H, x_k)

        _temp = np.linalg.inv( R + np.matmul( np.matmul( H, P_k), np.transpose(H)) )
        print("_temp", _temp)
        K_k = np.matmul( np.matmul(P_k, np.transpose(H)), _temp)

        print("K_k", K_k, y_k, x_k)
        bounding_box.predicted_state = x_k + np.matmul(K_k, y_k)

        print(bounding_box.predicted_state)
        _temp =np.eye(6) - np.matmul(K_k, H)
        
        print()
        bounding_box.predicted_error = np.matmul( np.matmul( _temp, P_k), np.transpose(_temp) ) + np.matmul( np.matmul(K_k, R),np.transpose(K_k) )

        
        print("bounding_box.predicted_error", bounding_box.predicted_error)
        # Update
        
        
        center_old = bounding_box.center
        x_hat = np.matmul(frame.F_MATRIX, bounding_box.predicted_state) 

        predicted_error =  np.matmul( np.matmul(frame.F_MATRIX, bounding_box.predicted_error) , np.transpose(frame.F_MATRIX) ) + frame.Q_MATRIX
        
        bounding_box.center = x_hat[:2]
        print("predicted_error", predicted_error, predicted_error.shape)
        print("center_old", center_old, center_old.shape)
        print("x_hat", bounding_box.center, x_hat.shape)
        print("_predict_next_frame_bounding_box")
        
        corners = bounding_box.get_corners() 
        print("corners", corners)
        
        #bounded_indices = bounding_box.filter_points(pc)
        
        new_bounding_box, pointsInside, corners = self.corners_to_bounding_box(corners, pc, False)
        
        new_bounding_box["predicted_error"] = np.diag(predicted_error).tolist()
        new_bounding_box["predicted_state"] = x_hat.tolist()
        
        
        print(new_bounding_box) 
        print("time to predict bounding box: ", time.time() - start)
            
        return new_bounding_box
        
  
    def calibrate_orientation(self, top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner):

        center = np.mean(np.vstack((top_right_corner, bottom_left_corner)), axis=0)
        w = np.linalg.norm(top_right_corner - top_left_corner)
        l = np.linalg.norm(top_left_corner[1] - bottom_left_corner[1])

        if w < l:
            w, l = l, w
            top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner = top_right_corner, bottom_right_corner, bottom_left_corner, top_left_corner 
            
            
        return top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner, center, w, l
    

    def corners_to_bounding_box(self, corners, points, is_shape_fitting_required=False, context=None):
        sorted_corners = sorted(corners, key=lambda x:x[1])
        if sorted_corners[2][0] > sorted_corners[3][0]:
            sorted_corners[2], sorted_corners[3] = sorted_corners[3], sorted_corners[2]
        if sorted_corners[0][0] > sorted_corners[1][0]:
            sorted_corners[0], sorted_corners[1] = sorted_corners[1], sorted_corners[0]

        top_right_corner = sorted_corners[3]
        top_left_corner = sorted_corners[2]
        bottom_left_corner = sorted_corners[0]
        bottom_right_corner = sorted_corners[1]
        
        
        top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner, center, w, l = self.calibrate_orientation( top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)
        
        
        top_right_corner = top_right_corner - top_left_corner
        angle = np.arctan2(top_right_corner[1], top_right_corner[0])
        top_right_corner += top_left_corner
        
        pointsInside = []
        
        if (is_shape_fitting_required):

            _origin = top_left_corner

            top_left_corner=top_left_corner-_origin
            top_right_corner=top_right_corner-_origin
            bottom_right_corner=bottom_right_corner-_origin
            bottom_left_corner=bottom_left_corner-_origin


            top_right_corner = top_right_corner - top_left_corner
            angle = np.arctan2(top_right_corner[1], top_right_corner[0])
            top_right_corner += top_left_corner



            points[:,:2] = points[:,:2] - _origin

            points[:,:2] = rotate_origin_only_bulk(points[:,:2], angle)


            #plt.scatter(points[:,1], points[:,0], c='r', s=1)

            #Rotate Point to Origin
            new_top_left_corner = rotate_origin_only(top_left_corner, angle)
            new_top_right_corner = rotate_origin_only(top_right_corner, angle)
            new_bottom_right_corner = rotate_origin_only(bottom_right_corner, angle)
            new_bottom_left_corner = rotate_origin_only(bottom_left_corner, angle)

            new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner, center, w, l = self.calibrate_orientation( new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner)


            #newline(top_left_corner,top_right_corner, 'g')
            #newline(top_right_corner,bottom_right_corner, 'g')
            #newline(bottom_right_corner,bottom_left_corner, 'g')
            #newline(bottom_left_corner,top_left_corner, 'g')


            #newline(new_top_left_corner,new_top_right_corner)
            #newline(new_top_right_corner,new_bottom_right_corner)
            #newline(new_bottom_right_corner,new_bottom_left_corner)
            #newline(new_bottom_left_corner,new_top_left_corner)


            #print("new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner, center, w, l")
            #print(new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner, center, w, l)


            search_number = 10

            car_size = CAR_SHAPE["suv"]


            _data_check = []
            for _y in np.linspace(new_bottom_left_corner[1], new_top_left_corner[1]-car_size["length"], search_number):
                for _x in np.linspace(new_bottom_left_corner[0], new_bottom_right_corner[0]-car_size["width"], search_number):


                    pointsInside = np.array(( 
                    points[:, 0] >= _x ,  points[:, 1] <= _x +car_size["width"] ,
                    points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

                    pointsInside = np.all(pointsInside , axis=0).nonzero()[0]

                    _data_check.append([_x, _y ,len(pointsInside)])



                    #top_left_corner = rotate_origin_only([_x, _y+car_size["length"]], -0)
                    #top_right_corner = rotate_origin_only([_x+car_size["width"], _y+car_size["length"]], -0)
                    #bottom_right_corner = rotate_origin_only([_x+car_size["width"], _y], -0)
                    #bottom_left_corner = rotate_origin_only([_x,_y], -0)


                    #newline(top_left_corner,top_right_corner, 'g')
                    #newline(top_right_corner,bottom_right_corner, 'g')
                    #newline(bottom_right_corner,bottom_left_corner, 'g')
                    #newline(bottom_left_corner,top_left_corner, 'g')


            _data_check = np.array( _data_check )

            #print(_data_check)

            _max = np.argmax(_data_check[:,2] )

            _x, _y = _data_check[_max,:2]

            pointsInside = np.array(( 
            points[:, 0] >= _x ,  points[:, 1] <= _x +car_size["width"] ,
            points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

            pointsInside = np.all(pointsInside , axis=0).nonzero()[0]



            #top_left_corner = rotate_origin_only([_x, _y+car_size["length"]], -0)
            #top_right_corner = rotate_origin_only([_x+car_size["width"], _y+car_size["length"]], -0)
            #bottom_right_corner = rotate_origin_only([_x+car_size["width"], _y], -0)
            #bottom_left_corner = rotate_origin_only([_x,_y], -0)



            #newline(top_left_corner,top_right_corner, 'r')
            #newline(top_right_corner,bottom_right_corner, 'r')
            #newline(bottom_right_corner,bottom_left_corner, 'r')
            #newline(bottom_left_corner,top_left_corner, 'r')



            top_left_corner = rotate_origin_only([_x, _y+car_size["length"]], -angle)
            top_right_corner = rotate_origin_only([_x+car_size["width"], _y+car_size["length"]], -angle)
            bottom_right_corner = rotate_origin_only([_x+car_size["width"], _y], -angle)
            bottom_left_corner = rotate_origin_only([_x,_y], -angle)




            top_left_corner=top_left_corner + _origin
            top_right_corner=top_right_corner +_origin
            bottom_right_corner=bottom_right_corner +_origin
            bottom_left_corner=bottom_left_corner + _origin

            top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner, center, w, l = self.calibrate_orientation( top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)


            newline(top_left_corner,top_right_corner, 'r')
            newline(top_right_corner,bottom_right_corner, 'r')
            newline(bottom_right_corner,bottom_left_corner, 'r')
            newline(bottom_left_corner,top_left_corner, 'r')


            #print("pointsInside", len(pointsInside) )
        
        
        if context:
            candidate_angles = np.array([angle-np.pi, angle, angle+np.pi])
            prev_angle = context.angle
            angle = candidate_angles[np.argmin(np.abs(candidate_angles - prev_angle))]


            

        bounding_box = {"center":center.tolist(), "angle":angle, "width":w, "length":l, 
                        "corner1":top_right_corner.tolist(), "corner2":bottom_left_corner.tolist()}, pointsInside, [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]

        #print(bounding_box)
        
        return bounding_box

    
    def predict_bounding_box(self, point, pc, settingsControls=None, num_seeds=5, plot=False, car_points=None, with_PCA=False):
        
        '''
        
        ActiveLearning: true
        AdaptiveCriterion: false
        Clustering: "DBSCAN"
        GroundRemoval: true
        OutlierRemoval: "RANSAC"
        ShapeFitting: false
        message: "Setting is no saved!"
        size: 0.5
        speed: 0.8

        '''
        search_range = float(settingsControls["SearchRange"])
        use_ground_removal = settingsControls["GroundRemoval"]
        type_criterion = int(settingsControls["FittingCriterion"])
        sampling_rate = settingsControls["SamplingRate"]
        outlier_removal = settingsControls["OutlierRemoval"]
        clustering_method = settingsControls["Clustering"]
        is_shape_fitting_required = settingsControls["ShapeFitting"]
        
        print("settingsControls", settingsControls);
        
        #png = pc
        print("point: {}".format(point))

        assert len(pc.shape) == 2, "pointcloud must have 2-dimensional shape"
        
        
        if (outlier_removal == "RANSAC" or outlier_removal == "OriginAwareRansac"): 
            number_of_iter = 200
            
        elif(outlier_removal == "None" or outlier_removal == "PCA"): 
            sampling_rate = 1.0
            number_of_iter = 1
            
        
        png = pc
        if( use_ground_removal):
            png = self.ground_plane_fitting(pc)["png"]
            
            
        
        if png.shape[1] == 4:
            png = png[:,:3]
        if point.size == 2:
            point = np.append(point, [0])
        if point.size == 4:
            point = point[:3]

        png[:,2] = 0
        
        
        if(clustering_method == "DBSCAN"):
            kd_tree = cKDTree(png)
            print(len(png))

            #trim png
            dists, ii = kd_tree.query(point, len(png))
            cutoff_idx = np.where(dists < 6)[0][-1]
            png_trimmed = png[ii[:cutoff_idx]]
            print(png_trimmed.shape)
            np.random.shuffle(png_trimmed)
            if png_trimmed.shape[0] > 5000:
                png_trimmed = png_trimmed[::4]
            elif png_trimmed.shape[0] > 2500:
                png_trimmed = png_trimmed[::2]
            kd_tree = cKDTree(png_trimmed)

            # Create random starting points for clustering algorithm
            std = .1
            seeds = np.random.randn(num_seeds, 3) * std + point
            seeds = np.vstack((point, seeds))

            dists, sample_indices = kd_tree.query(seeds)

            cluster_res = self.find_cluster(sample_indices, png_trimmed, th_dist=.5, num_nn=20, num_samples=20)
            png_source = cluster_res["cluster"]
        else:
            
            _dist = png[:,:2]-point[:2]
            
            _dist = np.sqrt( _dist[:,0:1]* _dist[:,0:1] + _dist[:,1:2]*_dist[:,1:2] )
            
            #print(_dist)
            indices_check = _dist <= search_range
            
            #print(indices_check.shape, len(indices_check.nonzero()[0]), png.shape)
            png_source = png[indices_check.nonzero()[0],:2]
            

        print(pc.shape, png.shape, png_source.shape)
     
        bbox_storage = []
        point_max_storage = []
        if plot:
            plt.style.use('dark_background');
            fig = plt.figure(figsize=(6,6))


        print("type_criterion", type_criterion)

        for __i in range(number_of_iter):
            


            rand_indices = np.random.choice(png_source.shape[0], int(png_source.shape[0]*sampling_rate))
            X = png_source[rand_indices]
        
            if( type_criterion == 0 ): 
                _criterion = area_criterion
            if( type_criterion == 1): 
                   _criterion = closeness_criterion
            elif(type_criterion == 3): 
                   _criterion = variance_criterion
          
            
            edges, corners = self.search_rectangle_fit(X, _criterion)
            

                
                
                
            bounding_box, pointsInside, corners = self.corners_to_bounding_box(corners, np.copy(png_source), is_shape_fitting_required)
            bbox_storage.append([bounding_box,pointsInside, corners ])
            point_max_storage.append( len(pointsInside) )
                   
                
        l_np = np.asarray(point_max_storage).argmax()
        
        #print(point_max_storage)
        
        corners =  np.array(bbox_storage[l_np][2])
        pointsInside = bbox_storage[l_np][1]
        bounding_box = bbox_storage[l_np][0]



        if plot:
            
            
            plt.ylim(( bounding_box['center'][0]+3,  bounding_box['center'][0]-3))
            plt.xlim(( bounding_box['center'][1]-3,  bounding_box['center'][1]+3))

            plt.scatter(png_source[pointsInside,1], png_source[pointsInside,0], c='#D4AF37', s=2.5)

            plt.scatter(png_source[:,1], png_source[:,0], c='r', s=1)
            plt.scatter(X[:,1], X[:,0], c='g', s=3)
            plt.scatter(corners[:,1], corners[:,0], c='#D4AF37', s=2) 
            self.plot_edges(corners)


            #plt.grid()
            fig.savefig('static/images/temp.png', dpi=fig.dpi)




        return bounding_box

    def plot_edges(self, corners, num_samples=100, c='#D4AF37', label=''):
        for i in range(4):
            v1, v2 = corners[i], corners[(i+1)%4]
            x = np.linspace(v1[0], v2[0], num_samples)
            y = np.linspace(v1[1], v2[1], num_samples)
            plt.plot(y, x, c=c, label=label)

    def search_farthest_nearest_neighbor(self, point, kd_tree, th_dist):
        num_nn = 2
        dists, nn_indices = kd_tree.query(point, num_nn)
        # print("th dist: ", th_dist)
        while (dists[-1] < th_dist):
            num_nn = num_nn * 2
            dists, nn_indices = kd_tree.query(point, num_nn)
        return dists, nn_indices

    def find_cluster(self, sample_indices, pc, th_dist=.2, density_thresh=10, num_nn=16, num_samples=20, overlap_thresh=.2):
        clusters = []
        seen_indices = []
        kd_tree = cKDTree(pc)
        for idx in sample_indices[:num_samples]:
            cluster = []
            queue = []
            seen = set()
            seen.add(idx)
            queue.append(idx)
            while len(queue):
                idx = queue.pop(0)
                point = pc[idx]
                cluster.append(point)
                dists, nn_indices = self.search_farthest_nearest_neighbor(point, kd_tree, th_dist)
                # dists, nn_indices = kd_tree.query(point, num_nn)
                if (len(nn_indices) > density_thresh):
                    for i in range(len(nn_indices)):
                        if nn_indices[i] not in seen and dists[i] < th_dist:
                            seen.add(nn_indices[i])
                            queue.append(nn_indices[i])
                
            clusters.append(np.vstack(cluster))
            seen_indices.append(np.array(list(seen)))
        
        overlapping_clusters = []
        # for i in range(len(seen_indices)):
        #     num_overlapping =  sum([len(np.intersect1d(seen_indices[i], seen_indices[j]))/len(seen_indices[i]) > overlap_thresh for j in range(len(seen_indices)) if j!=i])
        #     overlapping_clusters.append(num_overlapping)
        
        # largest_cluster = np.argmax(overlapping_clusters)
        # res = {"cluster": clusters[largest_cluster], "indices": seen_indices[largest_cluster]}

        # largest_cluster = np.unique(np.concatenate(seen_indices))
        largest_cluster = max(clusters, key=lambda cl:len(cl))
        res = {"cluster": largest_cluster, "indices": largest_cluster}
        return res
            

    def ground_plane_fitting(self, pc):
        x_max, x_min = np.max(pc[:,0]), np.min(pc[:,0])
        y_max, y_min = np.max(pc[:,1]), np.min(pc[:,1])
        seg_size_x = (x_max - x_min) / self.n_segs[0]
        seg_size_y = (y_max - y_min) / self.n_segs[1]
        res_pg = []
        res_png = []
        for i in range(self.n_segs[0]):
            for j in range(self.n_segs[1]):
                indices = np.intersect1d(np.intersect1d(np.where(pc[:,0] >= x_min + i*seg_size_x)[0], 
                                                        np.where(pc[:,0] < x_min + (i+1)*seg_size_x)[0]),
                                         np.intersect1d(np.where(pc[:,1] >= y_min + j*seg_size_y)[0], 
                                                        np.where(pc[:,1] < y_min + (j+1)*seg_size_y)[0]))
                if not len(indices):
                    continue
    #             print(len(indices))
                seg = pc[indices]
                pg = self.extract_initial_seeds(seg, self.n_lpr, self.th_seeds)
                png = []
                for _ in range(self.n_iter):
                    model = self.estimate_plane(pg)
                    pg, png = [], [np.zeros((1, 3))]
                    for p in seg:
                        if model(p) < self.th_dist:
                            pg.append(p)
                        else:
                            png.append(p)
    #                 print(len(pg), len(png))                    
                    pg, png = np.vstack(pg), np.vstack(png)
                    png = np.delete(png, 0, axis=0)
                res_pg.append(pg)
                res_png.append(png)
        res_pg = np.vstack(list(filter(len, res_pg)))
        res_png = np.vstack(list(filter(len, res_png)))
        res = {"pg": pg, "png": png}
        return res

    def extract_initial_seeds(self, pc, n_lpr, th_seeds):
        seeds = []
        psorted = np.sort(pc[:,2])
        LPR = np.mean(psorted[:self.n_lpr])
        for i in range(len(pc)):
            if pc[i,2] < LPR + self.th_seeds:
                seeds.append(pc[i])
        return np.vstack(seeds)

    def estimate_plane(self, pg):
        s_hat = np.mean(pg, axis=0)
        cov = sum([np.outer(s - s_hat, s - s_hat) for s in pg])
        u, s, vh = np.linalg.svd(cov, full_matrices=True)
        n = vh[2]
        d = -n @ s_hat
        def model(p):
            return abs((p - s_hat) @ n)
        return model
            
    def search_rectangle_fit(self, pc, criterion):
        pc = pc[:,:2]
        Q = dict()
        delta = np.pi / 180
        for theta in np.linspace(0, np.pi/2 - delta, 90*5):
            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])
            C1 = pc @ e1
            C2 = pc @ e2
            q = criterion(C1, C2)
            Q[theta] = q
        theta_star = max(Q.items(), key=lambda kv: kv[1])[0]
        # print(theta_star)
        C1_star = pc @ np.array([np.cos(theta_star), np.sin(theta_star)])
        C2_star = pc @ np.array([-np.sin(theta_star), np.cos(theta_star)])
        
        a1, b1, c1 = np.cos(theta_star), np.sin(theta_star), np.min(C1_star)
        a2, b2, c2 = -np.sin(theta_star), np.cos(theta_star), np.min(C2_star)
        a3, b3, c3 = np.cos(theta_star), np.sin(theta_star), np.max(C1_star)
        a4, b4, c4 = -np.sin(theta_star), np.cos(theta_star), np.max(C2_star)

        v1 = line_intersection(a1, b1, c1, a2, b2, c2)
        v2 = line_intersection(a2, b2, c2, a3, b3, c3)
        v3 = line_intersection(a3, b3, c3, a4, b4, c4)
        v4 = line_intersection(a1, b1, c1, a4, b4, c4)
        return [(a1, b1, c1), (a2, b2, c2), 
                (a3, b3, c3), (a4, b4, c4)], np.vstack([v1, v2, v3, v4])

def area_criterion(C1, C2):
    c1_max, c1_min = np.max(C1), np.min(C1)
    c2_max, c2_min = np.max(C2), np.min(C2)
    return -(c1_max-c1_min)*(c2_max-c2_min)
    
    
    
def line_intersection(a1, b1, c1, a2, b2, c2):
    x = (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
    y = (c1*a2 - c2*a1) / (b1*a2 - b2*a1)
    return np.array([x, y])

        
def variance_criterion(C1, C2):
    c1_max, c1_min = np.max(C1), np.min(C1)
    c2_max, c2_min = np.max(C2), np.min(C2)
    D1 = np.argmin([np.linalg.norm(c1_max - C1), np.linalg.norm(C1 - c1_min)])
    D2 = np.argmin([np.linalg.norm(c2_max - C2), np.linalg.norm(C2 - c2_min)])
    D1 = [c1_max - C1, C1 - c1_min][D1]
    D2 = [c2_max - C2, C2 - c2_min][D2]
    E1 = D1[np.where(D1 < D2)[0]]
    E2 = D2[np.where(D2 < D1)[0]]
    gamma = -np.var(E1) - np.var(E2)
    return gamma

def closeness_criterion(C1, C2, d=1e-4):
    c1_max, c1_min = np.max(C1), np.min(C1)
    c2_max, c2_min = np.max(C2), np.min(C2)
    D1 = np.argmin([np.linalg.norm(c1_max - C1), np.linalg.norm(C1 - c1_min)])
    D2 = np.argmin([np.linalg.norm(c2_max - C2), np.linalg.norm(C2 - c2_min)])
    D1 = [c1_max - C1, C1 - c1_min][D1]
    D2 = [c2_max - C2, C2 - c2_min][D2]
    beta = 0
    for i in range(len(D1)):
        d = max(min(D1[i], D2[i]), d)
        beta += 1/d
    return beta

# if __name__ == '__main__':
#   DATA_DIR = 'input/bin_data'
#   OUT_DIR = 'input/ground_removed'
#   bin_data  = sorted([f for f in listdir(DATA_DIR) 
#                       if isfile(join(DATA_DIR, f)) and '.bin' in f])

#   frame_names = [f.split(".")[0] for f in bin_data]
#   print(frame_names)
#   fh = FrameHandler()
#   bp = BoundingBoxPredictor(fh)
#   # fname1 = '0000000000'
#   # fname2 = '0000000001'

#   # frame1 = fh.load_annotation(fname1)
#   # frame2 = fh.load_annotation(fname2)

#   # print(bp.predict_next_frame_bounding_boxes(frame2))
#   for fname in frame_names:
#       read_filename = join(DATA_DIR, fname.split(".")[0] + ".bin")
#       data = np.fromfile(read_filename, dtype=np.float32)
#       data = data.reshape((-1,4))[:,:3]
#       print('input shape: {}'.format(data.shape))
#       output = bp.ground_plane_fitting(data)['png']
#       output = np.hstack((output, np.zeros((len(output), 1))))
#       print('output shape: {}'.format(output.shape))
#       save_filename = join(OUT_DIR, fname.split(".")[0] + ".bin")
#       output.astype(np.float32).tofile(save_filename)




