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
from models import NextFrameBBOX, is_overlap_with_other_boxes, homogeneous_transformation

from sklearn.cluster import DBSCAN
from mask_rcnn import get_pointcnn_labels


object_types = {'Pedestrian':1, 'Car': 2, 'Cyclist': 3, 'Truck': 4,     'Person_sitting' : 5, 'Motorbike' : 6, 'Trailer' : 7, 'Bus' : 8, 'Railed' : 9, 'Airplane' : 10, 'Boat' : 11, 'Animal' :12, 'DontCare' : 13, 'Misc' : 14, 'Van' : 15, 'Tram' : 16, 'Utility' : 17}
    
object_types_reverse = {}
for key, value in object_types.items():
    object_types_reverse[value] = key.lower()
print("object_types_reverse", object_types_reverse)

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
             pretty(value, indent+1)
        else:
             print('\t' * (indent+1) + str(value))

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


def filter_pointcloud(bbox, pointcloud):
    theta = bbox["angle"]
    transformed_pointcloud = homogeneous_transformation(pointcloud, bbox["center"], -theta)
    indices = np.intersect1d(np.where(np.abs(transformed_pointcloud[:,0]) <= bbox["width"]/2)[0], 
                             np.where(np.abs(transformed_pointcloud[:,1]) <= bbox["length"]/2)[0])
    return np.delete(pointcloud, indices, axis=0), pointcloud[indices,:]


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

    
    
    def fully_automated_bbox(self, fname, json_request):
        
        
        drivename, fname = fname.split("/")

        idx = self.frame_handler.drives[drivename].index(fname)       
        
        ground_removed  = False
    
        foreground_class = get_pointcnn_labels(drivename+"/"+fname, json_request["settingsControls"], ground_removed=ground_removed)

        
        point_classes_indices =  get_pointcnn_labels(drivename+"/"+fname, json_request["settingsControls"], ground_removed=ground_removed, foreground_only=False)
        
        
        #print("foreground_class", foreground_class)
        #print("point_classes_indices", point_classes_indices)
        
        
        pc = self.frame_handler.get_pointcloud(drivename, fname, dtype=float, ground_removed=ground_removed)
        
        points_class = pc[foreground_class]
        point_classes_indices = point_classes_indices[foreground_class]
        
        
        max_distance_per_class =1.0
        type_criterion =  2
        is_shape_fitting_required = False
               
        
        #object_ids, center_cluster = matrix_scan(points_class[ : ,:3], max_distance_per_class, 20)
        
        
        #Calculate euclidian to the center:
        
        #norm_density =  np.sqrt((points_class[ : ,0] ** 2 + points_class[ : ,1] ** 2 ), axis=0)
        #print("norm_density", norm_density.shape)
        

        
        clustering = DBSCAN(eps=max_distance_per_class, min_samples=100, metric='euclidean').fit(points_class[ : ,:2])
        
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
            object_class = point_classes_indices[individual_object_indices]
            
            unique_class, number_point_per_class = np.unique(object_class, return_counts=True)
            
            object_name_idx = unique_class[np.argmax(number_point_per_class)]
            object_name = object_types_reverse[object_name_idx]
            print("object_class", object_name, np.unique(object_class, return_counts=True))

            centroid = [np.mean(png_source[:,0]), np.mean(png_source[:,1]), np.min(png_source[:,2])]


            X = png_source

            edges, corners = self.search_rectangle_fit(X, closeness_criterion)


            bounding_box, pointsInside, corners = self.corners_to_bounding_box(corners, np.copy(png_source), is_shape_fitting_required)

            bounding_boxes_opt[str(object_id)] = bounding_box
            bounding_boxes_opt[str(object_id)]["object_id"] = object_name
                

        return bounding_boxes_opt
    
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
        
        
    def predict_next_frame_bounding_boxes(self, frame, json_request):
        
        fh = FrameHandler()
        
        main_start = time.time() 

        drivename, fname = frame.fname.split('.')[0].split("/")

        idx = self.frame_handler.drives[drivename].index(fname)
        next_fname = self.frame_handler.drives[drivename][idx+1]
        current_fname = self.frame_handler.drives[drivename][idx]
        
        
        
        #prev_frame = fh.load_annotation(drivename, current_fname, json_request["settingsControls"])

        
        self.next_frame_idx = idx+1    
        _search_number = 30
        self.search_number = _search_number
        self.treshold_point_annotation_error = 1
        _padding = 0.1
        self.padding = _padding
        self.max_angle_changes_on_deg = 15

        
        ground_removed  = json_request["settingsControls"]["GroundRemoval"]
        is_guided_tracking  = json_request["settingsControls"]["GuidedTracking"]
        
        car_points = get_pointcnn_labels(drivename+"/"+next_fname, json_request["settingsControls"], ground_removed=ground_removed)
        
        
        #current_pc = self.frame_handler.get_pointcloud(drivename, current_fname, dtype=float, ground_removed=ground_removed)
        
        #current_pc_png = np.copy(self.ground_plane_fitting(current_pc)['png'])
        
        
        
        next_all_pc_fresh = self.frame_handler.get_pointcloud(drivename, next_fname, dtype=float, ground_removed=ground_removed)
        
        next_pc = np.copy(next_all_pc_fresh[car_points])
        
        z_axis = np.max(next_pc[:,2]) # Ground already removed
        
        next_all_pc = np.copy(next_all_pc_fresh[next_all_pc_fresh[:,2]> z_axis-3 ])
        
        
        
        #print(fname, ground_removed)
        #print([box.box_id for box in frame.bounding_boxes])
        bounding_boxes = sorted(frame.bounding_boxes, 
                            key=lambda box:box.box_id)
        centers = {box.box_id:box.center for box in bounding_boxes}
        velocities = {box_id:np.zeros(2) for box_id in centers.keys()}
        

        next_bounding_boxes = [] 
        #Init bounding boxes
        print("Init bounding boxes", self.padding, self.search_number, self.treshold_point_annotation_error, is_guided_tracking )
        
        self.prevBbox = {}
        self.boxHistoryLocations = {}
        F = frame.F_MATRIX
        for bounding_box in bounding_boxes:
            
            self.prevBbox[bounding_box.box_id] = {}
            
            self.prevBbox[bounding_box.box_id]["max_distance"] = 5
            
            self.boxHistoryLocations[bounding_box.box_id] = {} #Init history location
            
            start = time.time()
            self.padding = _padding
            self.search_number = _search_number
            _pc = next_pc
            
           
            print("box id", bounding_box.box_id, bounding_box.tracking_idx)
            if bounding_box.tracking_idx > 1:
                self.padding = 0.05 
                self.search_number = 20
                
                #bounding_box.grow_pointcloud(np.copy(next_all_pc))
                
                
                
                x_hat_k =  bounding_box.predicted_state
                center_old = bounding_box.center
                x_hat_k_prediction = np.matmul(F, x_hat_k)  # + Q should be N(0, Q) 

                x_diff = np.abs(x_hat_k-x_hat_k_prediction)
                update_length = np.max(x_diff[:2])

                self.prevBbox[bounding_box.box_id]["max_distance"] = update_length + 1
                #prev_pc_annotated_bbox = bounding_box.filter_pointcloud( np.copy(current_pc_png), update_length*2 )[1]
                
                pc_annotated_bbox = bounding_box.filter_pointcloud( np.copy(next_pc), update_length*2 )[1]
                
                if(pc_annotated_bbox.shape[0] <= 300): #Use check bbox
                    _, self.prevBbox[bounding_box.box_id]["pcs"] = bounding_box.filter_pointcloud( np.copy(next_all_pc),1.0)
                    bounding_box.center = x_hat_k_prediction[:2]
                    _, pred_pcs = bounding_box.filter_pointcloud( np.copy(next_all_pc),1.0)
                    
                    self.prevBbox[bounding_box.box_id]["pcs"] = np.concatenate( [ pred_pcs, np.copy(self.prevBbox[bounding_box.box_id]["pcs"]) ], axis=0)
                    bounding_box.center = center_old
                    
                else:
                    self.prevBbox[bounding_box.box_id]["pcs"] = np.copy(next_pc) #pc_annotated_bbox
                        
                
                print("\t\t prev_pc_annotated_bbox", update_length, pc_annotated_bbox.shape, self.prevBbox[bounding_box.box_id]["pcs"].shape)
            else:
                
                _, pc_annotated_bbox = bounding_box.filter_pointcloud( np.copy(_pc), 0.0 )
                if(pc_annotated_bbox.shape[0] > 100):                     
                    self.prevBbox[bounding_box.box_id]["pcs"] = np.copy(_pc)
                else:
                    _, self.prevBbox[bounding_box.box_id]["pcs"] = bounding_box.filter_pointcloud( np.copy(next_all_pc), 1.0 ) 
                
                
            _pc = np.concatenate( [ np.copy(_pc), np.copy(self.prevBbox[bounding_box.box_id]["pcs"]) ], axis=0)
                
            new_bbox = self._predict_next_frame_bounding_box(frame, bounding_box, np.copy(_pc), is_guided_tracking) 
            next_bounding_boxes.append( NextFrameBBOX(bounding_box.box_id, new_bbox[1], new_bbox[0],  new_bbox[2], bounding_box.tracking_idx) )

            print("time to predict bounding box: ", bounding_box.box_id, time.time() - start)
  


        self.next_bounding_boxes = next_bounding_boxes

        #Clean overlapping boxes 
        if( is_guided_tracking):

            start = time.time()
            try:
                self.fixed_overlapping_boxes(False)
            except:
                pass
        print("time to fixed_overlapping_boxes: ", time.time() - start)


        
        _padding = 0.2       
        self.treshold_point_annotation_error = 1
        print("Retrack bounding boxes", self.padding, self.search_number, self.treshold_point_annotation_error )
        #Retrack box locations 
        
        is_bboxes_updated = np.ones([len((self.next_bounding_boxes)) ])
        
        is_bboxes_updated[:] = True
        self.search_number = 20
        if( is_guided_tracking):   
            while( is_bboxes_updated.any() and _padding > 0.01 ):               
                
        
                #is_bboxes_updated[:] = True

                updated_bounding_boxes = []          


                print("self.padding", start, self.padding)
                
                self.padding = _padding
                idx_box_status = 0
                for bbox in self.next_bounding_boxes:

                    
                    start = time.time()
                    print("box id", bbox.box_id)
                    box_state = bbox.get_bounding_box({})   
                    all_corners_set = {}
                    all_corners_set[bbox.get_tracking_index()] = [bbox.get_corners(), bbox.get_center_dist()] # Init location

                    
                    _pc = next_pc
            

                    if bbox.tracking_idx > 1:
                        self.padding = _padding  * 1/bbox.tracking_idx
                        self.search_number = 20
                        _pc = np.concatenate( [ np.copy(_pc), np.copy(self.prevBbox[bbox.box_id]["pcs"]) ], axis=0)



                    if(is_bboxes_updated[idx_box_status]):
                    

                        _, all_corners_set = self.guided_search_bbox_location(bbox.get_corners(), np.copy(_pc), bbox.get_tracking_index(), all_corners_set, bbox)             

                    updated_bounding_boxes.append( NextFrameBBOX(bbox.box_id, all_corners_set, box_state, bbox.center, bbox.tracking_idx) )

                    print("time to predict bounding box: ", bbox.box_id, time.time() - start)

                    
                    idx_box_status = idx_box_status +1


                print("Retrack box locations: ", time.time() - start)


                self.next_bounding_boxes = updated_bounding_boxes

                #Clean overlapping boxes 
                
                start = time.time()
                try:
                    self.fixed_overlapping_boxes(False)
                except:
                    pass
                print("time to fixed_overlapping_boxes: ", time.time() - start)
                
                idx_box_status = 0
                for bbox_check in self.next_bounding_boxes:
                    is_bboxes_updated[idx_box_status] = bbox_check.is_bbox_updated
                    idx_box_status = idx_box_status +1
                    
                _padding = _padding * 0.1
                
                print("is_bboxes_updated",is_bboxes_updated,  is_bboxes_updated.any() )
                 

                
        final_bounding_boxes = {}
        criterion = closeness_criterion
        
        
        start = time.time()

        for bbox in self.next_bounding_boxes:
            
            
            
            bbox_structure, _, _ = self.corners_to_bounding_box( bbox.get_corners(), None, False)
            final_bounding_boxes[str(bbox.id)]=bbox.get_bounding_box(bbox_structure)
            

    
            #bbox_structure["angle"] = new_angle
            
            """
            #Update angle if required
            _, pcInside = filter_pointcloud(bbox_structure, np.copy(next_all_pc[:,:2]) )
            #print("pcInside", pcInside.shape)
            #_, new_corners = self.search_rectangle_fit(pcInside, criterion, False)
            
            
            #print("new_corners", new_corners.shape, new_corners)
            
            #new_bbox_structure, _, _ = self.corners_to_bounding_box(new_corners, None, False)
            
            update_angle =  self.search_rectangle_fit(pcInside, criterion, True)
            old_angle = math.degrees(bbox_structure["angle"])
            new_angle = math.degrees(update_angle)
            
  
            print(str(bbox.id), "new_angle", abs( old_angle - new_angle ), old_angle, new_angle)
        
            if(abs( old_angle - new_angle )  < self.max_angle_changes_on_deg):
                final_bounding_boxes[str(bbox.id)]["angle"] = update_angle

            """
        print("Recalculate angle: ", time.time() - start)
        
        #pretty( self.boxHistoryLocations )
        print("Total time for predict frame: ", time.time() - main_start)
        return final_bounding_boxes, round(time.time() - main_start,2)
    
    
    
    def fixed_overlapping_boxes(self, is_overlap_exist):

        for bbox_check in self.next_bounding_boxes:
            for bbox in self.next_bounding_boxes:
                if(bbox_check.id != bbox.id):
                    cur_index = -1
                    while(bbox_check.is_boxes_overlap(bbox) and cur_index!= 0): #IF the two boxes still overlap
                        #print(cur_index, bbox_check.id, bbox.id)
                        is_overlap_exist = True
                        if(bbox_check.get_center_dist() >=  bbox.get_center_dist()):
                            cur_index = bbox_check.update_index()
                        else:
                            cur_index = bbox.update_index()
                
        if(is_overlap_exist):
            return self.fixed_overlapping_boxes(False)
        
        
    def _predict_next_frame_bounding_box(self, frame, bounding_box, pc, is_guided_tracking):
        """Pure state to state linear movement Kalman Filter"""
        
        
        box_state = {}
        box_state["object_id"] = bounding_box.object_id
        box_state["predicted_error"] = np.diag(bounding_box.predicted_error).tolist()
        box_state["predicted_state"] = bounding_box.predicted_state.tolist()
        box_state["tracking_idx"] = bounding_box.tracking_idx + 1

        
        
        
        F = frame.F_MATRIX
        x_hat_k =  bounding_box.predicted_state
        
        
        center_old = bounding_box.center
        x_hat_k_prediction = np.matmul(F, x_hat_k)  # + Q should be N(0, Q) 
        
        
        if(bounding_box.tracking_idx > 3):
            _,_, v_x, v_y, _,_ = box_state['predicted_state']
            if(v_x == 0):
                v_x = 0.0000001
            updated_angle = math.atan(v_y/v_x)        

            #updated_angle = math.atan2(v_y, v_x)
            
            old_angle = math.degrees( bounding_box.angle)
            new_angle = math.degrees(updated_angle)
            
  
            if( abs( old_angle - new_angle ) < 5 ):
                
                print(str(bounding_box.object_id), "angle updated! [diff, old, new]", abs( old_angle - new_angle ), old_angle, new_angle)
                bounding_box.angle = updated_angle
            
            
        
        bounding_box.center = x_hat_k_prediction[:2]
        print("x_hat_k_prediction", x_hat_k_prediction)

        
            
        # Updating new corners with x_hat (new prediction state) as center
        corners = bounding_box.get_corners() 
        
        """BBOX guided greedy update with Point annotation"""
        
        all_corners_set = {}
        
        
        _, pc_annotated_bbox = bounding_box.filter_pointcloud( np.copy(pc), 0)
        number_of_points_init = pc_annotated_bbox.shape[0]
                
        all_corners_set[number_of_points_init] = [corners, 0.0] # Init location
        
        if(is_guided_tracking):        
            corners, all_corners_set = self.guided_search_bbox_location(corners, pc, number_of_points_init, all_corners_set, bounding_box)
            #print("all_corners_set", all_corners_set)
   
        return box_state, all_corners_set, bounding_box.center
    """
    def filter_pointcloud(self, pointcloud, obj_shape, updated_size = 0):
        angle, center, width, length = obj_shape
        theta = angle
        transformed_pointcloud = homogeneous_transformation(pointcloud,
                center, -theta)
        indices = \
            np.intersect1d(np.where(np.abs(transformed_pointcloud[:,
                           0]) <= (width+updated_size) / 2)[0],
                           np.where(np.abs(transformed_pointcloud[:,
                           1]) <= (length+updated_size) / 2)[0])
        return (np.delete(pointcloud, indices, axis=0),
                pointcloud[indices, :])
    
    def guided_search_bbox_optimized(self, corners, points, max_points, all_corners_set, bounding_box):
        
        center_predict = bounding_box.center
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
        
        
        #print("-start", top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)
        
        top_right_corner = top_right_corner - top_left_corner
        angle = np.arctan2(top_right_corner[1], top_right_corner[0])
        top_right_corner += top_left_corner
        
        
        search_number = self.search_number # 100*100 search space

        car_size = {"width": w, "length": l}
        
        history_locations = {}

        _data_check = []
        
        ys = np.linspace(bottom_left_corner[1]-(car_size["length"]*self.padding), top_left_corner[1]+(car_size["length"]*self.padding), search_number)
        xs = np.linspace(bottom_left_corner[0]-(car_size["width"]*self.padding), bottom_right_corner[0]+(car_size["width"]*self.padding), search_number)

        for _y in ys:
            if( _y not in history_locations):
                history_locations[_y] = {}
            for _x in xs:
                
                if( _x not in history_locations[_y]):
                    obj_shape = [angle, center[0]+ ]
                    _, pc_inside = self.filter_pointcloud(self, np.copy(pc), obj_shape, updated_size = 0)
                    
                

                top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner = self.rotate_corners( _x, _y, car_size, angle, _origin)


                
                
        
    """    
    def guided_search_bbox_location(self, corners, points, max_points, all_corners_set, bounding_box):
        
        
        center_predict = bounding_box.center
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
        
        print("\tcorners", max_points, center, points.shape, "padding", self.padding, self.search_number)
        
        #print("-start", top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner)
        
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
        search_number = self.search_number # 100*100 search space

        car_size = {"width": w, "length": l}

        _data_check = []
        
        ys = np.linspace(new_bottom_left_corner[1]-(car_size["length"]*self.padding), new_top_left_corner[1]+(car_size["length"]*self.padding), search_number)
        xs = np.linspace(new_bottom_left_corner[0]-(car_size["width"]*self.padding), new_bottom_right_corner[0]+(car_size["width"]*self.padding), search_number)

        for _y in ys:
            
            for _x in xs:
                
                _key = rotate_origin_only([_x,_y], -angle)
                
                x_key =float("{0:.3f}".format(_key[0] + _origin[0])) 
                y_key =float("{0:.3f}".format( _key[1] + _origin[1]))  
                
                if x_key not in self.boxHistoryLocations[bounding_box.box_id]:
                    self.boxHistoryLocations[bounding_box.box_id][x_key] = {}
                
                if( y_key not in self.boxHistoryLocations[bounding_box.box_id][x_key] ):

                    top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner = self.rotate_corners( _x, _y, car_size, angle, _origin)

                    corner_checks= np.vstack([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])


                    center = np.mean(np.vstack((top_right_corner, bottom_left_corner)), axis=0)

                    dist = np.sqrt( np.sum( np.square( center[:2] -  center_predict[:2]) ))
                    
                    #print(dist, self.prevBbox[bounding_box.box_id]["max_distance"],  center)
                    if( dist < self.prevBbox[bounding_box.box_id]["max_distance"]): 


                        self.boxHistoryLocations[bounding_box.box_id][x_key][y_key] = 0
                        if( True or is_overlap_with_other_boxes(bounding_box.box_id, corner_checks, self.next_bounding_boxes) == False ):

                            pointsInside = np.array(( 
                            points[:, 0] >= _x ,  points[:, 0] <= _x +car_size["width"] ,
                            points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

                            pointsInside = np.all(pointsInside , axis=0).nonzero()[0]

                            if(len(pointsInside) > 10):

                                _data_check.append([_x, _y ,len(pointsInside), pointsInside])
                                self.boxHistoryLocations[bounding_box.box_id][x_key][y_key] = len(pointsInside)




        
        if(len(_data_check)> 0):
            _data_check = np.array( _data_check )

            _best_location = np.argmax(_data_check[:,2] )

            _number_of_points = _data_check[_best_location,2]


            error_annotation_treshold = self.treshold_point_annotation_error
                

            if( max_points >= (_number_of_points-error_annotation_treshold) ):

                print("\t\tnp.max(_data_check[:,2] ) < max_points", _number_of_points, max_points )
                return corners, all_corners_set
            
            else:

            

                # select max with error treshold
                idx_max_all = (_data_check[:,2] >= (_number_of_points - error_annotation_treshold) ).nonzero()[0] 

                # If multiple locations have the same number of contained points 
                if(len(idx_max_all) > 1): 
                    #Get bounding box location by minimazing all-point distance to 4 box-edges
                    rest = np.ones(len(idx_max_all) ) * 999999

                    dist_arg_min_storage = []
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
                        dist_arg_min = np.argmin(dist_all, axis=0)

                        dist_min = np.amin(dist_all, axis=0)

                        #print("dist_arg_min", dist_all.shape, dist_arg_min.shape, dist_min.shape)
                        
                        #dist_arg_min = dist_arg_min[ dist_min < 0.5]

                        rest[idx_loop_max] =np.mean(dist_min) # np.var(dist_arg_min) #
                        dist_arg_min_storage.append(dist_arg_min)
                        idx_loop_max = idx_loop_max +1

                    _min = np.argmin(rest) # get location with minimum point distances

                    for __i in range(len(dist_arg_min_storage)):


                        top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner = self.rotate_corners( _data_check[idx_max_all[__i],0], _data_check[idx_max_all[__i],1], car_size, angle, _origin)
                        corners= np.vstack([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])

                        center = np.mean(np.vstack((top_right_corner, bottom_left_corner)), axis=0)


                        print("_storage", _min, __i, "rest", rest[__i],  np.unique(dist_arg_min_storage[__i], return_counts=True), "location", center )


                    print("dist_arg_min_storage", np.unique(dist_arg_min_storage[_min], return_counts=True) )
                    _best_location = idx_max_all[_min] # Update current best point location


                _x, _y = _data_check[_best_location,:2]

                pointsInside = np.array(( 
                points[:, 0] >= _x ,  points[:, 0] <= _x +car_size["width"] ,
                points[:, 1] >= _y ,  points[:, 1] <= _y +car_size["length"] ) )

                pointsInside = np.all(pointsInside , axis=0).nonzero()[0]


                top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner = self.rotate_corners( _x, _y, car_size, angle, _origin)


                corners= np.vstack([top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner])

                center = np.mean(np.vstack((top_right_corner, bottom_left_corner)), axis=0)

                dist = np.sqrt( np.sum( np.square( center - center_predict ) ))
                if(_number_of_points > 10): # and bounding_box.tracking_idx < 3):
                    all_corners_set[_number_of_points] = [corners, dist]

                    if(_number_of_points > max_points):

                        #print("guided_search_bbox_location", _number_of_points, corners.shape)
                        return self.guided_search_bbox_location(corners, pc, _number_of_points, all_corners_set, bounding_box)

                print("_number_of_points < max_points ", _number_of_points , max_points )
        return corners, all_corners_set
           
           
   
    def rotate_corners(self, _x, _y, car_size, angle, _origin):
        
        
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

        
        return top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner
        
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

    
    def predict_bounding_box(self, point, pc_all, settingsControls=None, num_seeds=5, plot=False, car_points=None, with_PCA=False):
        
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
        if(car_points != None):
        
            pc= np.copy(pc_all[car_points])
        else:
            pc= np.copy(pc_all)
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

        
        
        if(clustering_method == "DBSCAN"):
            png[:,2] = 0
            """
            kd_tree = cKDTree(png)
            print(len(png))

            #trim png
            dists, ii = kd_tree.query(point, len(png))
            cutoff_idx = np.where(dists < 8)[0][-1]
            png_trimmed = png[ii[:cutoff_idx]]
            print(png_trimmed.shape)
            np.random.shuffle(png_trimmed)
            if png_trimmed.shape[0] > 5000:
                png_trimmed = png_trimmed[::4]
            elif png_trimmed.shape[0] > 2500:
                png_trimmed = png_trimmed[::2]
                
            
            png[:,2] = 0
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

            
            """
            """
            _dist = png[:,:2]-point[:2]
            
            _dist = np.sqrt( _dist[:,0:1]* _dist[:,0:1] + _dist[:,1:2]*_dist[:,1:2] )
            
            indices_check = _dist <= search_range # NN search
            png = png[indices_check.nonzero()[0],:]
            z_axis = np.min(png[:,2]) # Ground already removed
            
                        
            png_all_fresh = np.copy(pc_all) #self.ground_plane_fitting(np.copy(pc_all))["png"]
            
            _dist_all = png_all_fresh[:,:2]-point[:2]
            
            _dist_all = np.sqrt( _dist_all[:,0:1]* _dist_all[:,0:1] + _dist_all[:,1:2]*_dist_all[:,1:2] )
            
            indices_check_all = _dist_all <= search_range # NN search
            
            
            png_all_fresh = png_all_fresh[indices_check_all.nonzero()[0],:]  #Filtered NN all
            
            print("z_axis", z_axis, np.min(png[::,2]))
            
            png_all_pc = np.copy(png_all_fresh[png_all_fresh[:,2]> z_axis+0.1 ])
        
            """
            png_trimmed = png #np.concatenate( [ png, png_all_pc ], axis=0)
                        
            print("png_trimmed", png_trimmed.shape)
            kd_tree = cKDTree(png_trimmed)

            std = .1
            seeds = np.random.randn(num_seeds, 3) * std + point
            seeds = np.vstack((point, seeds))

            dists, sample_indices = kd_tree.query(seeds)
            

            
            
        
            png_trimmed = np.concatenate( [ png_trimmed, seeds ], axis=0)
            
            seed_cluster_idx = -1
            
            min_samples=30 
            max_eps=.45 
            while seed_cluster_idx == -1:

                clustering = DBSCAN(eps=max_eps, min_samples=min_samples, metric='euclidean').fit(png_trimmed[ : ,:2])

                object_ids = clustering.labels_

                seed_cluster_idx = object_ids[-1]
                
                print("png_trimmed", png_trimmed.shape, min_samples, max_eps, len(object_ids == seed_cluster_idx), object_ids[-3], object_ids[-2], object_ids[-1], np.unique(object_ids,return_counts=True))
                
                #min_samples = min_samples * 0.9
                max_eps = max_eps * 1.1
                



            png_source = png_trimmed[ object_ids == seed_cluster_idx ,:2]
            
            
            
            #cluster_res = self.find_cluster(sample_indices, png_trimmed, th_dist=.5, num_nn=20, num_samples=20)
            #png_source = cluster_res["cluster"]
        else:
            
        
            png[:,2] = 0
            _dist = png[:,:2]-point[:2]
            
            _dist = np.sqrt( _dist[:,0:1]* _dist[:,0:1] + _dist[:,1:2]*_dist[:,1:2] )
            
            indices_check = _dist <= search_range # NN search
            png = png[indices_check.nonzero()[0],:]
            
            """
            z_axis = np.min(png[:,2]) # Ground already removed
            
                        
            png_all_fresh = np.copy(pc_all) #self.ground_plane_fitting(np.copy(pc_all))["png"]
            
            _dist_all = png_all_fresh[:,:2]-point[:2]
            
            _dist_all = np.sqrt( _dist_all[:,0:1]* _dist_all[:,0:1] + _dist_all[:,1:2]*_dist_all[:,1:2] )
            
            indices_check_all = _dist_all <= search_range # NN search
            
            
            png_all_fresh = png_all_fresh[indices_check_all.nonzero()[0],:]  #Filtered NN all
            
            print("z_axis", z_axis, np.min(png[::,2]))
            
            png_all_pc = np.copy(png_all_fresh[png_all_fresh[:,2]> z_axis+0.1 ])
        
        
            png = np.concatenate( [ png, png_all_pc ], axis=0)
            
            #print(indices_check.shape, len(indices_check.nonzero()[0]), png.shape)
            """
            png_source = png #png[indices_check.nonzero()[0],:2]
            

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
            
    def search_rectangle_fit(self, pc, criterion, angle_only = False):
        pc = pc[:,:2]
        Q = dict()
        delta = np.pi / 180
            
        for theta in np.linspace(0, np.pi/2 - delta, 90*2):
            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])
            C1 = pc @ e1
            C2 = pc @ e2
            q = criterion(C1, C2)
            Q[theta] = q
        theta_star = max(Q.items(), key=lambda kv: kv[1])[0]
        
        if(angle_only):
            return theta_star
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




