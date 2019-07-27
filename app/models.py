import numpy as np


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
