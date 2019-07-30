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
from models import NextFrameBBOX

from mask_rcnn import get_pointcnn_labels

def distances_points_to_line(p0, p1, p2):
     
    #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    n = p0.shape[0]
    x0= p0[:,0]
    y0= p0[:,1]
    x1= np.ones(n) * (p1[0])
    y1= np.ones(n) * (p1[1])
    x2= np.ones(n) * (p2[0])
    y2= np.ones(n) * (p2[1])
    
    return np.abs( ((y2-y1)*x0) - ((x2-x1)*y0) + (x2*y1) - (y2*x1) ) / np.sqrt( np.square(y2-y1) + np.square(x2-x1 )  )


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
        self.next_bounding_boxes = []
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
        
        car_points = get_pointcnn_labels(drivename+"/"+next_fname, json_request["settingsControls"], ground_removed=ground_removed)
        
        
        next_pc = self.frame_handler.get_pointcloud(drivename, next_fname, dtype=float, ground_removed=ground_removed)
        
        next_pc = next_pc[car_points]
        
        
        #print(fname, ground_removed)
        #print([box.box_id for box in frame.bounding_boxes])
        bounding_boxes = sorted(frame.bounding_boxes, 
                            key=lambda box:box.box_id)
        centers = {box.box_id:box.center for box in bounding_boxes}
        velocities = {box_id:np.zeros(2) for box_id in centers.keys()}
        
        """
        next_pc[:,2] = 0
        next_pc = next_pc[:,:3]
        np.random.shuffle(next_pc)
        next_pc_small = next_pc[::4]
        
        """
        self.next_bounding_boxes = [] #Init bounding boxes
        for bounding_box in bounding_boxes:
            
            start = time.time()

            new_bbox = self._predict_next_frame_bounding_box(frame, bounding_box, np.copy(next_pc)) 
            self.next_bounding_boxes.append( NextFrameBBOX(bounding_box.box_id, new_bbox[1], new_bbox[0]) )
            
            #Clean overlapping boxes  
            self.fixed_overlapping_boxes(False)

            print("time to predict bounding box: ", bounding_box.box_id, time.time() - start)


                    
        final_bounding_boxes = {}
        for bbox in self.next_bounding_boxes:
            
            bbox_structure, _, _ = self.corners_to_bounding_box( bbox.get_corners(), None, False)
            final_bounding_boxes[str(bbox.id)]=bbox.get_bounding_box(bbox_structure)
        

        return final_bounding_boxes
    
    


    
    def fixed_overlapping_boxes(self, is_overlap_exist):

        for bbox_check in self.next_bounding_boxes:
            for bbox in self.next_bounding_boxes:
                if(bbox_check.id != bbox.id):
                    cur_index = -1
                    while(bbox_check.is_boxes_overlap(bbox) and cur_index!= 0): #IF the two boxes still overlap
                        print(cur_index, bbox_check.id, bbox.id)
                        is_overlap_exist = True
                        if(bbox_check.get_center_dist() >=  bbox.get_center_dist()):
                            cur_index = bbox_check.update_index()
                        else:
                            cur_index = bbox.update_index()
                
        if(is_overlap_exist):
            return self.fixed_overlapping_boxes(False)
        
        
    def _predict_next_frame_bounding_box(self, frame, bounding_box, pc):
        """Pure state to state linear movement Kalman Filter"""
        
        # Previous state initialization
        z_k = bounding_box.center
        x_k =  bounding_box.predicted_state
        P_k =  bounding_box.predicted_error
      
        if(np.sum(x_k[:2])== 0):
            x_k[:2] = z_k
            
        H = frame.H_MATRIX
        R = frame.R_MATRIX

        y_k = z_k - np.matmul(H, x_k)
        
        _temp = np.linalg.inv( R + np.matmul( np.matmul( H, P_k), np.transpose(H)) )
        K_k = np.matmul( np.matmul(P_k, np.transpose(H)), _temp)
        bounding_box.predicted_state = x_k + np.matmul(K_k, y_k)


        _temp =np.eye(6) - np.matmul(K_k, H)
        bounding_box.predicted_error = np.matmul( np.matmul( _temp, P_k), np.transpose(_temp) ) + np.matmul( np.matmul(K_k, R),np.transpose(K_k) )

        # Update operation
        center_old = bounding_box.center
        x_hat = np.matmul(frame.F_MATRIX, bounding_box.predicted_state) 
        predicted_error =  np.matmul( np.matmul(frame.F_MATRIX, bounding_box.predicted_error) , np.transpose(frame.F_MATRIX) ) + frame.Q_MATRIX        
        bounding_box.center = x_hat[:2]

        kalman_state = {}
        kalman_state["predicted_error"] = np.diag(predicted_error).tolist()
        kalman_state["predicted_state"] = x_hat.tolist()

        
        # Updating new corners with x_hat (new prediction state) as center
        corners = bounding_box.get_corners() 
        
        """BBOX guided greedy update with Point annotation"""
        
        all_corners_set = {}
        all_corners_set[-1] = [corners, 0.0] # Init location
        
        corners, all_corners_set = self.guided_search_bbox_location(corners, pc, -1, all_corners_set, bounding_box.center)
        #new_bounding_box, pointsInside, corners = self.corners_to_bounding_box(corners, pc, False)
        
            
        return kalman_state, all_corners_set
        
        
    def guided_search_bbox_location(self, corners, points, max_points, all_corners_set, center_predict):
        
        pc = np.copy(points) #Backup original
        
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
        
        
        print("-start", top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)
        
        top_right_corner = top_right_corner - top_left_corner
        angle = np.arctan2(top_right_corner[1], top_right_corner[0])
        top_right_corner += top_left_corner
        
        #Centerized all point to origin (0,0)
        _origin = top_left_corner

        top_left_corner=top_left_corner-_origin
        top_right_corner=top_right_corner-_origin
        bottom_right_corner=bottom_right_corner-_origin
        bottom_left_corner=bottom_left_corner-_origin


        top_right_corner = top_right_corner - top_left_corner
        angle = np.arctan2(top_right_corner[1], top_right_corner[0])
        top_right_corner += top_left_corner
        
        points[:,:2] = points[:,:2] - _origin
        
        
        #Rotate Points to 0 degree
        points[:,:2] = rotate_origin_only_bulk(points[:,:2], angle)

        
        new_top_left_corner = rotate_origin_only(top_left_corner, angle)
        new_top_right_corner = rotate_origin_only(top_right_corner, angle)
        new_bottom_right_corner = rotate_origin_only(bottom_right_corner, angle)
        new_bottom_left_corner = rotate_origin_only(bottom_left_corner, angle)

        new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner, center, w, l = self.calibrate_orientation( new_top_left_corner, new_top_right_corner, new_bottom_right_corner, new_bottom_left_corner)

        
        #Search location with maximum number of contained points
        search_number = 100 # 100*100 search space

        car_size = {"width": w, "length": l}

        _data_check = []
        
        ys = np.linspace(new_bottom_left_corner[1]-(car_size["length"]/5), new_top_left_corner[1]+(car_size["length"]/5), search_number)
        xs = np.linspace(new_bottom_left_corner[0]-(car_size["width"]/5), new_bottom_right_corner[0]+(car_size["width"]/5), search_number)

        for _y in ys:
            for _x in xs:
                pointsInside = np.array(( 
                points[:, 0] >= _x ,  points[:, 0] <= _x +car_size["width"] ,
                points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

                pointsInside = np.all(pointsInside , axis=0).nonzero()[0]

                _data_check.append([_x, _y ,len(pointsInside), pointsInside])



        _data_check = np.array( _data_check )

        _best_location = np.argmax(_data_check[:,2] )
        
        _number_of_points = _data_check[_best_location,2]
        
        error_annotation_treshold = 10
        
        # select max with error treshold
        idx_max_all = (_data_check[:,2] >= (_number_of_points - error_annotation_treshold) ).nonzero()[0] 
        
        # If multiple locations have the same number of contained points 
        if(len(idx_max_all) > 1): 
            #Get bounding box location by minimazing all-point distance to 4 box-edges
            rest = np.ones(len(idx_max_all) ) * 999999
            
            idx_loop_max = 0
            for idx_bbox in idx_max_all:  
                _x, _y, _, pointsInside = _data_check[idx_bbox,:]
               
                
                p0 = np.copy( points[pointsInside, :])
                c1 = np.array([_x, _y])
                c2 = np.array([_x, _y+l])
                c3 = np.array([_x+w, _y])
                c4 = np.array([_x+w, _y+l])
            
                d1 = distances_points_to_line(p0, c1, c2)
                d2 = distances_points_to_line(p0, c1, c3)
                d3 = distances_points_to_line(p0, c3, c4)
                d4 = distances_points_to_line(p0, c4, c2)
            
                dist_all = np.vstack([d1,d2,d3,d4])
                dist_min = np.min(dist_all, axis=0)
                
                rest[idx_loop_max] = np.mean(dist_min)
                idx_loop_max = idx_loop_max +1
           
            _min = np.argmin(rest) # get location with minimum point distances
            
            _best_location = idx_max_all[_min] # Update current best point location
        
            
        _x, _y = _data_check[_best_location,:2]

        pointsInside = np.array(( 
        points[:, 0] >= _x ,  points[:, 0] <= _x +car_size["width"] ,
        points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

        pointsInside = np.all(pointsInside , axis=0).nonzero()[0]


        # Recovering corner orientation, with angle rotation!

        top_left_corner = rotate_origin_only([_x, _y+car_size["length"]], -angle)
        top_right_corner = rotate_origin_only([_x+car_size["width"], _y+car_size["length"]], -angle)
        bottom_right_corner = rotate_origin_only([_x+car_size["width"], _y], -angle)
        bottom_left_corner = rotate_origin_only([_x,_y], -angle)




        # Recovering point location, with + _origin box location!
        top_left_corner=top_left_corner + _origin
        top_right_corner=top_right_corner +_origin
        bottom_right_corner=bottom_right_corner +_origin
        bottom_left_corner=bottom_left_corner + _origin

        corners= np.vstack([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])
        
        center = np.mean(np.vstack((top_right_corner, bottom_left_corner)), axis=0)
        
        dist = np.sqrt( np.sum( np.square( center - center_predict ) ))
        all_corners_set[_number_of_points] = [corners, dist]
            
        if(_number_of_points > max_points ):
            
            print("guided_search_bbox_location", _number_of_points, corners.shape)
            return self.guided_search_bbox_location(corners, pc, _number_of_points, all_corners_set, center_predict)
        
        return corners, all_corners_set
           
           
   
            
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
            

        if(png_source.shape[0] < 4):
            return ""
        
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




