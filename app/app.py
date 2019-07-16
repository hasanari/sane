from flask import Flask, render_template, request, jsonify, send_from_directory
from models import BoundingBox
from pointcloud import PointCloud
from predict_label import predict_label
from mask_rcnn import get_pointcnn_labels
from frame_handler import FrameHandler
from bounding_box_predictor import BoundingBoxPredictor
import numpy as np
import json
import os
from pathlib import Path

app = Flask(__name__, static_url_path='/static')
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

@app.route("/")
def root():
	return render_template("index.html")

@app.route("/initTracker", methods=["POST"])
def init_tracker():
	json_request = request.get_json()
	pointcloud = PointCloud.parse_json(json_request["pointcloud"])
	tracker = Tracker(pointcloud)
	return "success"

@app.route("/trackBoundingBoxes", methods=['POST'])
def trackBoundingBox():
	json_request = request.get_json()
	pointcloud = PointCloud.parse_json(json_request["pointcloud"], json_request["intensities"])
	filtered_indices = tracker.filter_pointcloud(pointcloud)
	next_bounding_boxes = tracker.predict_bounding_boxes(pointcloud)
	print(next_bounding_boxes)
	return str([filtered_indices, next_bounding_boxes])


@app.route("/updateBoundingBoxes", methods=['POST'])
def updateBoundingBoxes():
	json_request = request.get_json()
	bounding_boxes = BoundingBox.parse_json(json_request["bounding_boxes"])
	tracker.set_bounding_boxes(bounding_boxes)
	return str(bounding_boxes)



@app.route("/predictLabel", methods=['POST'])
def predictLabel():
	json_request = request.get_json()
	json_data = json.dumps(json_request)
	filename = json_request['filename'].split('.')[0]
	os.system("rm {}/*".format(os.path.join(DIR_PATH, "static/images")))
	predicted_label = predict_label(json_data, filename)
	in_fov = os.path.exists(os.path.join(DIR_PATH, "static/images/cropped_image.jpg"))
	return ",".join([str(predicted_label), str(in_fov)])


@app.route("/getMaskRCNNLabels", methods=['POST'])
def getMaskRCNNLabels():
	json_request = request.get_json()
	filename = json_request['fname']
	return str(get_pointcnn_labels(filename, json_request["settingsControls"]))


@app.route("/writeOutput", methods=['POST'])
def writeOutput():
    
	frame = request.get_json()['output']
	settingsControls = frame['settingsControls']
	fname = frame['filename']
	drivename, fname = fname.split('/')
#	if(settingsControls["FrameTracking"]):
	fh.save_annotation(drivename, fname, frame["file"], settingsControls)
	return str("hi")


@app.route("/loadFrameNames", methods=['POST'])
def loadFrameNames():
	return fh.get_frame_names()

@app.route("/getFramePointCloud", methods=['POST'])
def getFramePointCloud():
	json_request = request.get_json()
	fname = json_request["fname"]
	drivename, fname = fname.split("/")
	data_str = fh.get_pointcloud(drivename, fname, dtype=str)
	annotation_str = str(fh.load_annotation(drivename, fname, json_request["settingsControls"], dtype='json'))
	return '?'.join([data_str, annotation_str])

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(app.root_path, 'favicon.ico')

@app.route("/predictBoundingBox", methods=['POST'])
def predictBoundingBox():
	json_request = request.get_json()
	fname = json_request["fname"]
	drivename, fname = fname.split("/")
	point = json_request["point"]
	point = np.array([point['z'], point['x'], point['y']])
    
	print("Points", point, json_request)
	ground_removed  = False
    
	car_points = get_pointcnn_labels(json_request["fname"], json_request["settingsControls"], ground_removed=ground_removed)
    
	# frame = fh.get_pointcloud(drivename, fname, dtype=float, ground_removed=False)
	# print("num points with ground: {}".format(frame.shape))
	frame = fh.get_pointcloud(drivename, fname, dtype=float, ground_removed=ground_removed)
	return str(bp.predict_bounding_box(point, frame[car_points], settingsControls=json_request["settingsControls"]))

@app.route("/predictNextFrameBoundingBoxes", methods=['POST'])
def predictNextFrameBoundingBoxes():
	json_request = request.get_json()
	fname = json_request["fname"]
	drivename, fname = fname.split("/")
	frame = fh.load_annotation(drivename, fname, json_request["settingsControls"])
	if(frame == ""):
		return "error"
	else: 
		res = bp.predict_next_frame_bounding_boxes(frame, json_request)
		keys = res.keys()
		for key in keys:
			res[str(key)] = res.pop(key)
		print(res)
		return str(res)


@app.route("/fully_automated_bbox", methods=['POST'])
def fully_automated_bbox():
	json_request = request.get_json()
	fname = json_request["fname"]
	res = bp.fully_automated_bbox(fname, json_request)
	keys = res.keys()
	for key in keys:
		res[str(key)] = res.pop(key)
	print(res)

	return str(res)


if __name__ == "__main__":
	fh = FrameHandler()
	bp = BoundingBoxPredictor(fh)
	os.system("rm {}/*".format(os.path.join(DIR_PATH, "static/images")))
	app.run(host='0.0.0.0', port=33000)
