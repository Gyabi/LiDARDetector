import argparse
import csv
from hashlib import new
from time import sleep
import cv2
import numpy as np
import os
import torch
import pdb

from utils import setup_seed, read_points, read_calib, read_label, \
    keep_bbox_from_image_range, keep_bbox_from_lidar_range, vis_pc, \
    vis_img_3d, bbox3d2corners_camera, points_camera2image, \
    bbox_camera2lidar
from model import PointPillars

def main(params):
    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
    LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
    pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)

    if not params["no_cuda"]:
        model = PointPillars(nclasses=len(CLASSES)).cuda()
        model.load_state_dict(torch.load(params["ckpt"]))
    else:
        model = PointPillars(nclasses=len(CLASSES))
        model.load_state_dict(
            torch.load(params["ckpt"], map_location=torch.device('cpu')))
    
    if not os.path.exists(params["pc_path"]):
        raise FileNotFoundError 
    pc = read_mypoint(params["pc_path"])
    pc_torch = torch.from_numpy(pc)
    
    model.eval()
    with torch.no_grad():
        if not params["no_cuda"]:
            pc_torch = pc_torch.cuda()
    
        import time
        time1 = time.time()
        result_filter = model(batched_pts=[pc_torch], 
                              mode='test')[0]
        print("{}".format(time.time()-time1))
    
    # csv出力    
    output_csv(result_filter, params["pc_path"].split("/")[-1].split(".")[0]+"_detect"+".csv")
    
    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
    lidar_bboxes = result_filter['lidar_bboxes']
    labels, scores = result_filter['labels'], result_filter['scores']

    vis_pc(pc, bboxes=lidar_bboxes, labels=labels)
            
def read_mypoint(path):    
    datas = np.loadtxt(path, encoding="utf-8_sig", delimiter=",", dtype="float32")
    return datas

# csv出力
def output_csv(result, filename):
    with open("output/"+filename, "w", newline="") as f:
        writer = csv.writer(f)
        for (bbox, label) in zip(result["lidar_bboxes"], result["labels"]):
            data = bbox.tolist()
            data.append(label)
            writer.writerow(data)    
        
if __name__ == '__main__':    
    
    params = {
        "no_cuda" : False,
        "ckpt" : "pretrained/epoch_160.pth",
        "pc_path" : "C:/Users/buyuu/UnityProjects/001_制作物/LiDARObjectDetectionSim/LiDARObjectDetectionSim/Assets/StreamingAssets/LiDAR/20221010231713.csv"
    }
    main(params)
