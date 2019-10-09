

# _SAnE_: Smart annotation and evaluation tools for point cloud data

### Abstract
Bridging the needs to provide high-quality, time-efficient, and easy-to-use annotation tools, we propose SAnE, a semi-automatic annotation tool for point cloud data. While most current methods rely on multi-sensor approaches to provide bounding box annotations, we here focus on maximizing the potentials of point cloud data alone to provide high-quality point cloud labels. Our contributions of this paper are threefold: (1) we propose a denoising pointwise segmentation strategy enabling one-click annotation technique, (2) we expand the motion model technique with our novel guided-tracking algorithm easing the frame-to-frame annotation process, and (3) we provide an interactive yet robust open-source point cloud annotation tool simplifying the creation of high-quality bounding box annotations. Using KITTI dataset, we show that our approach speeds up the annotation process by a factor of 4.17 while achieving IoU agreements of 92.02\% and 82.22\% for 2D bounding box (BBOX) and Bird Eye View (BEV), respectively. A more careful annotation even achieves +19.37\% higher IoU agreement than the KITTI IoU agreement with the ground truth data.

A demonstration of SAnE can be found below (at 2x speed):

![Alt Text](https://github.com/hasanari/smart-annotation/blob/master/sane-point-cloud.gif)

## Installation
1. Clone this repository
2. Setup virtual environment:
   ```Shell
   conda create -n sane python=3.6 anaconda
   ```
   Activate the virtual environment
   ```Shell
   source activate sane
   ```
3. Install dependencies. By default we use Python3.6 for SAnE and PointCNN.
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained denoising weights (denoising_weights.zip) from the [releases page](https://drive.google.com/open?id=1Uysbfz_4cdl9BQAYHBUBw7wCs_zZ6SNA) into pointcnn-models/denoise. The file structure should be:
	```bash
	sane/pointcnn-models/denoise/pretrained.data-00000-of-00001
	sane/pointcnn-models/denoise/pretrained.index
	sane/pointcnn-models/denoise/pretrained.meta
   ```
 4. (Optional) Install and test PointCNN sampling module.
	   ```bash
	   cd app/PointCNN/sampling/
	   sh tf_sampling_compile.sh
	   python tf_sampling.py
	   
	   #Change sampling module at Line[60] in app/PointCNN/pointcnn_seg/kitti3d_x8_2048_fps.py
	   from -> sampling = 'random' 
	   to -> sampling = 'fps' 
	   ```
6. To run the tool, run `python app.py` in wherever you have your `app` directory is.
8. Open http://127.0.0.1:7772/ on a browser.


### Troubleshooting

 - CUDA version mismatch

>   Try to install CUDA version 9.++ with cuDNN v7.05, SAnE uses tensorflow v1.7.0 which was compiled with that cuDNN v7.05.

 - Failed to compile PointCNN sampling module

>  Check here https://github.com/yangyanli/PointCNN/issues/87

## Acknowledgment

Most of the codes and code structures are taken from Latte annotation tool (https://github.com/bernwang/latte/) and PointCNN pointwise segmentation (https://github.com/yangyanli/PointCNN).

**Changes and Updates**:
1. Automatic detection algorithm:
> - Denoising PointCNN with adaptive-sampling approach.
> - PointCNN for object-classification.
> - PointCNN with bin-based regression.

2. Refined interface features and functionalities:
> - Removing dependency on ground removal.
> - Simplifying object fitting process.
> - Adding outlier removal algorithm.
> - Speed up clustering process.
> - Providing background-level annotation.
> - Addressing object occluded problem.

3. Updated visuals:
> - 3D bounding box view.
> - Added more control.
> - Top view per selected object.

