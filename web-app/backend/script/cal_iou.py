
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import cv2
from itertools import groupby
import argparse
from pathlib import Path

def _extractBbox(result,old_ratio = 1024,new_ratio = 640):
  groupby(result, key = lambda x:x['image_id'])
  list_result = [list(v) for k,v in groupby(result, key = lambda x:x['image_id'])]
  bbox_list = []

  for im in list_result:

    bbox = [list(map( lambda x: int((x/old_ratio)*new_ratio) , box['bbox'])) for box in im]
    bbox_list.append(bbox)
  return bbox_list

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="input 2 number")
  parser.add_argument('pred_path', type=str, help='path of prediction' )
  parser.add_argument('val_path', type=str, help='path of validation' )
  args = parser.parse_args()
  
  result_path = args.pred_path
  val_path = args.val_path
#   print(result_path, val_path)
  # result_path = 'crcnn_x101_result_coco.json'
  # val_path = 'val_dataset.json'
  
  width = 1024
  heigh = 1024
  global new_dim 
  new_dim = 640

  # Opening JSON file
  ground_truth_file = open(val_path)
  ground_truth = json.load(ground_truth_file)['annotations']
  gt_boxes = _extractBbox(ground_truth)
  len(gt_boxes[0])

  with open(result_path , 'r') as f:
    result = json.load(f)
  rs_boxes = _extractBbox(result)
  len(rs_boxes[0])


  # making canvas
  iou = []
  for i in range(len(rs_boxes)):
      
      gt_img = np.zeros([new_dim,new_dim])
      rs_img = np.zeros([new_dim,new_dim])
      for box in rs_boxes[i]:
          start_point = (box[0],box[1])
          end_point = (box[0]+box[2],box[1]+box[3])
          cv2.rectangle(rs_img, start_point, end_point, (1,1), -1)
          
      for box in gt_boxes[i]:
          start_point = (box[0],box[1])
          end_point = (box[0]+box[2],box[1]+box[3])
          cv2.rectangle(gt_img, start_point, end_point, (1,1), -1)
          
      flat_gt = gt_img.flatten()
      flat_rs = rs_img.flatten()
      flat_sum = flat_gt + flat_rs

      unique, counts = np.unique(flat_sum, return_counts=True)
      couting_val = dict(zip(unique, counts))

      intersection = couting_val[2.0]
      union = couting_val[1.0] + couting_val[2.0]

      iou.append(intersection/union)
      
  mean_iou = round(np.mean(iou),3)
  print(f"The mean IoU = {mean_iou}")






