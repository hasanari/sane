import numpy as np

def range_overlap(a_min, a_max, b_min, b_max):
	'''Neither range is completely greater than the other
	'''
	return (a_min <= b_max) and (b_min <= a_max)

        
class NextFrameBBOX():
	def __init__(self, box_id, back_tracking_boxes, kalman_state):
		self.id = box_id
		self.back_tracking_boxes = back_tracking_boxes
		self.kalman_state = kalman_state
		self.box_track_indices = sorted(back_tracking_boxes.keys())
		self.current_box_track_index = len(self.box_track_indices)-1 # from very last
        
	def update_index(self):
		self.current_box_track_index = self.current_box_track_index - 1
		if(self.current_box_track_index < 0):
			self.current_box_track_index = 0
		return self.current_box_track_index
    
	def get_tracking_index(self):
		return self.box_track_indices[self.current_box_track_index]
        
	def get_corners(self):
		return self.back_tracking_boxes[self.get_tracking_index()][0]
    
	def get_center_dist(self):
		return self.back_tracking_boxes[self.get_tracking_index()][1]
    
	def get_bounding_box(self, bbox):

		bbox["center_dist"] = self.get_center_dist()
		bbox["predicted_state"] = self.kalman_state["predicted_state"] 
		bbox["predicted_error"] = self.kalman_state["predicted_error"] 
		return bbox

	def is_boxes_overlap(self, box_check):
		return self.overlap(self.get_corners(), box_check.get_corners())
    
    
        
	#https://codereview.stackexchange.com/questions/31352/overlapping-rectangles
	def overlap(self, corners_new, corners):
		'''Overlapping rectangles overlap both horizontally & vertically
		'''

		r1 = {'left': np.min(corners[:,0]), 'right': np.max(corners[:,0]), 'bottom': np.min(corners[:,1]), 'top': np.max(corners[:,1]) }
		r2 = {'left': np.min(corners_new[:,0]), 'right': np.max(corners_new[:,0]), 'bottom': np.min(corners_new[:,1]), 'top': np.max(corners_new[:,1]) }

		return range_overlap(r1["left"], r1["right"], r2["left"], r2["right"]) and range_overlap(r1["bottom"], r1["top"], r2["bottom"], r2["top"])

    

class Frame():
	def __init__(self, fname, bounding_boxes, dt=0.1):
		self.fname = fname
		self.bounding_boxes = bounding_boxes
		self.F_MATRIX =  np.array([[1, 0, dt, 0, 0.5*dt*dt, 0], 
                        [0, 1, 0, dt, 0, 0.5*dt*dt],
                         [0, 0, 1, 0, dt, 0],
                         [0, 0, 0, 1, 0, dt],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]
                        ])
		self.Q_MATRIX = np.eye(6) * [0.1, 0.1, 1, 1, 10, 10]
		self.R_MATRIX = np.eye(2) * [0.001, 0.001]
		self.H_MATRIX =  np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
        

	@staticmethod
	def parse_json(json_frame):
		json_bounding_boxes = json_frame['frame']['bounding_boxes']
		bounding_boxes = BoundingBox.parse_json(json_bounding_boxes)
		return Frame(json_frame['frame']['fname'], bounding_boxes)

class BoundingBox():
	def __init__(self, box_id, center, width, length, angle, object_id, predicted_state, predicted_error, settingsControls):
		self.box_id = box_id
		self.x = center['x']
		self.y = center['y']
		self.center = np.array([self.x, self.y])
		self.width = width
		self.length = length
		self.angle = angle
		self.settingsControls = settingsControls
		self.object_id = object_id
		self.predicted_error = np.eye(6) * np.array(predicted_error)
		self.predicted_state = np.transpose( np.array(predicted_state) )

	@staticmethod
	def parse_json(json):
		return [BoundingBox(json_obj['box_id'],
							json_obj['center'], 
							json_obj['width'],
							json_obj['length'],
							json_obj['angle'],
							json_obj['object_id'],
							json_obj['predicted_state'],
							json_obj['predicted_error'],
							json_obj['settingsControls'])
							 for json_obj in json]

	def filter_points(self, pointcloud, bounding_factor=.1):
		l, w, theta = self.length, self.width, self.angle
		center = np.array([[self.x, self.y]])
		rotated_points = pointcloud.rigid_transform(theta, center)
		x, y = rotated_points[:, 0], rotated_points[:, 1]
		indices_within_width = np.where(np.abs(x) <= w / 2 * (1 + bounding_factor))[0]
		indices_within_length = np.where(np.abs(y) <= l / 2 * (1 + bounding_factor))[0]

		bounded_indices = np.intersect1d(indices_within_width, indices_within_length)
		return bounded_indices
	
	def filter_pointcloud(self, pointcloud):
		theta = self.angle
		transformed_pointcloud = homogeneous_transformation(pointcloud, self.center, -theta)
		indices = np.intersect1d(np.where(np.abs(transformed_pointcloud[:,0]) <= self.width/2)[0], 
								 np.where(np.abs(transformed_pointcloud[:,1]) <= self.length/2)[0])
		return np.delete(pointcloud, indices, axis=0), pointcloud[indices,:]
	
	def get_corners(self):
		c1 = np.array([-self.width/2, -self.length/2])
		c2 = np.array([self.width/2, -self.length/2])
		c3 = np.array([self.width/2, self.length/2])
		c4 = np.array([-self.width/2, self.length/2])
		corners = homogeneous_transformation(np.vstack([c1, c2, c3, c4]), np.zeros(2), self.angle) + self.center
		return corners

def homogeneous_transformation(points, translation, theta):
	return (points[:,:2] - translation).dot(rotation_matrix(theta).T)

def rotation_matrix(theta):
	return np.array([[np.cos(theta), -np.sin(theta)], 
					 [np.sin(theta), np.cos(theta)]])
