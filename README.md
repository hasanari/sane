

# _SAnE_: Smart annotation and evaluation tools for point cloud data

### Abstract
Addressing the need for high-quality, time-efficient, and easy-to-use annotation tools, we propose SAnE, a semi-automatic annotation tool for labeling point cloud data. While most current methods rely on multi-sensor approaches to provide bounding box annotations, we focus on the potential of the point cloud data itself for providing high-quality labelling in an efficient way. The contributions of this paper are threefold: (1) we propose a denoising pointwise segmentation strategy enabling a fast implementation of one-click annotation, (2) we expand the motion model technique with our guided-tracking algorithm, easing the frame-to-frame annotation processes, and (3) we provide an interactive yet robust open-source point cloud annotation tool, targeting both skilled and crowdsourcing annotators to create high-quality bounding box annotations. Using the KITTI dataset, we show that the SAnE speeds up the annotation process by a factor of 4.44 while achieving Intersection over Union (IoU) agreements of 84.27%. Furthermore, in experiments using crowdsourcing services, the full-featured SAnE achieves an accuracy of 79.36% while reducing the annotation time by a factor of 3, a significant improvement compared to the baseline accuracy of 62.02%. This result shows the potential of AI-assisted annotation tools, such as SAnE, for providing fast and accurate annotation labels for large-scale datasets with a significantly reduced price.

A demonstration of SAnE can be found below (at 3x speed):

![SAnE](https://github.com/hasanari/sane/blob/develop/sane-point-cloud_.gif)

For more details, please refer to our paper: https://ieeexplore.ieee.org/document/9143095. If you find this work useful for your research, please consider citing:

H. A. Arief et al., "SAnE: Smart Annotation and Evaluation Tools for Point Cloud Data," in IEEE Access, vol. 8, pp. 131848-131858, 2020, doi: 10.1109/ACCESS.2020.3009914.



``` 
@ARTICLE{9143095,
  author={H. A. {Arief} and M. {Arief} and G. {Zhang} and Z. {Liu} and M. {Bhat} and U. G. {Indahl} and H. {Tveite} and D. {Zhao}},
  journal={IEEE Access}, 
  title={SAnE: Smart Annotation and Evaluation Tools for Point Cloud Data}, 
  year={2020},
  volume={8},
  number={},
  pages={131848-131858},
  keywords={Three-dimensional displays;Tools;Noise reduction;Crowdsourcing;Two dimensional displays;Proposals;Robustness;Annotation tool;crowdsourcing annotation;frame tracking;point cloud data},
  doi={10.1109/ACCESS.2020.3009914},
  ISSN={2169-3536},
  month={},}
   ```
   
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

>   Try to install CUDA version 9.++ with cuDNN v7.05, SAnE uses tensorflow v1.7.0 which was compiled with cuDNN v7.05.

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
