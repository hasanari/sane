import os
import sys
import numpy as np
import os.path
import time

python27 = " CUDA_VISIBLE_DEVICES=0 /home/mansur/anaconda3/envs/tf_p36/bin/python3.6 "
script_axcrf = " PointCNN/predicting_point_segmentation_with_axcrf.py  "
script_seg = "  PointCNN/predicting_point_segmentation.py "


def get_pointcnn_labels_axcrf(filename):
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    drivename, fname = filename.split("/")    
    if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"axcrf.bin") ) == False :

        os.system(" killall python3.6 & ")
            
        os.system(python27+ script_axcrf + "--retrieve_whole_files=0 --filename={}".format(filename))
        
        os.system(python27+ script_axcrf + "--retrieve_whole_files=1 --filename={}".format(filename)+ " &")

    bounded_indices = np.fromfile(
                        os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"axcrf.bin"),
                        dtype=np.int)
    #os.system("rm {}/PointCNN/output/"+drivename+"_"+fname+".bin".format(ROOT_DIR))

    return bounded_indices.tolist()
    # os.system("rm classify/bounding_boxes/*.json")

def get_pointcnn_labels(filename, settingsControls, ground_removed=False):
    
    
    print("please wait....", filename)

    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    drivename, fname = filename.split("/")    

    
    if(settingsControls["WithDenoising"] == False):
        postfix = "normal-weights"

        if(ground_removed):
            
            if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"ground_removed.bin") ) == False :

                os.system(" killall python3.6 & ")

                os.system(python27+ script_seg + " --ground_removed=1 --retrieve_whole_files=0 --filename={}".format(filename))

                os.system(python27+ script_seg + " --ground_removed=1 --retrieve_whole_files=1 --filename={}".format(filename)+ " &")
            
            
            bounded_indices = np.fromfile(
                            os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"ground_removed.bin"),
                            dtype=np.int)
        
        
        else: #Non-Ground Removed
            
            if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin") ) == False :

                os.system(" killall python3.6 & ")

                os.system(python27+ script_seg + " --ground_removed=0 --retrieve_whole_files=0 --postfix="+postfix+" --filename={}".format(filename))


                os.system(python27+ script_seg + " --ground_removed=0  --retrieve_whole_files=1 --postfix="+postfix+" --filename={}".format(filename)+ " & ")
                
                
            bounded_indices = np.fromfile(
                os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"),
                dtype=np.int)
            
            

    else: # Denoise
        
        postfix = "denoise-weights"
        
        
        if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin") ) == False :

            os.system(" killall python3.6 & ")

            os.system(python27+ script_seg + " --ground_removed=0 --retrieve_whole_files=0 --postfix="+postfix+" --filename={}".format(filename))
            
            
            os.system(python27+ script_seg + " --ground_removed=0  --retrieve_whole_files=1 --postfix="+postfix+" --filename={}".format(filename)+ " & ")
            
            
        bounded_indices = np.fromfile(
                            os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"),
                            dtype=np.int)
    
    return bounded_indices.tolist()

