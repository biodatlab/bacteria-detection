from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import cv2
import mmcv
import pandas as pd
from mmdet.apis import init_detector, inference_detector
import yaml
from loguru import logger


PRJ_ROOT = Path(__file__).parents[3] / "web-app"
CONFIG_DIR = PRJ_ROOT / "backend" / "config.yaml"

with open(CONFIG_DIR, "r") as f:
    config = yaml.safe_load(f)
DEVICE = config['device']


def _init_model(model_dir: str, check_point_dir: str, device=DEVICE):
    logger.info(f'using device = {DEVICE}')
    config_file = model_dir
    check_point = check_point_dir
    model = init_detector(config_file, check_point, device=device)
    return model

def mmdet_to_coco(detection_results, image_id):
    annotations = []
    for label, det_result in enumerate(detection_results):
        for i in range(det_result.shape[0]):
            x1, y1, x2, y2, score = det_result[i]
            w, h = x2 - x1, y2 - y1
            annotation = {
                'id': len(annotations) + 1,
                'image_id': image_id,
                'category_id': label, # replace with the category ID of the detected object
                'bbox': list(map(lambda x: int(x),[x1, y1, w, h])),
                'score': score.astype('float64')
            }
            annotations.append(annotation)
    return annotations

def coco_to_csv(coco_json: List[Dict[str, str]]):
    annotation = coco_json['annotations']
    image = coco_json['images']
    category = coco_json['categories']
    
    annotation_df = pd.DataFrame(annotation)
    image_df = pd.DataFrame(image)
    image_df['image_id'] = image_df['id']
    category_df = pd.DataFrame(category)
    category_df['category_id'] = category_df['id']
    
    annotation_df = pd.merge(annotation_df, image_df, on='image_id')
    annotation_df = pd.merge(annotation_df,category_df, on='category_id')

    coco_csv = pd.DataFrame()
    coco_csv['file_name'] = annotation_df['file_name']
    coco_csv['width'] = annotation_df['width']
    coco_csv['height'] = annotation_df['height']
    coco_csv['class'] = annotation_df['name']
    coco_csv['xmin'] = annotation_df['bbox'].apply(lambda x: x[0])
    coco_csv['ymin'] = annotation_df['bbox'].apply(lambda x: x[1])
    coco_csv['xmax'] = annotation_df['bbox'].apply(lambda x: x[0] + x[2])
    coco_csv['ymax'] = annotation_df['bbox'].apply(lambda x: x[1] + x[3])
    coco_csv['score'] = annotation_df['score']
    
    return coco_csv.to_csv()
    

def get_result(image: np.ndarray, model_dir:str, check_point_dir: str, image_id: int):
    model = _init_model(
        model_dir,
        check_point_dir
    )
    image = image
    result = inference_detector(model, image)
    coco_result = mmdet_to_coco(result, image_id)


    return coco_result


def coco_to_mmdet(annotations):
    num_classes = max([ann['category_id'] for ann in annotations]) + 1
    detection_results = [[] for _ in range(num_classes)]
    
    for ann in annotations:
        label = ann['category_id']
        bbox = ann['bbox']
        score = ann['score']
        
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        
        detection_result = [x1, y1, x2, y2, score]
        detection_results[label].append(detection_result)
    
    detection_results = [np.array(det_result) if det_result != []
                         else np.empty(shape=(0,5), dtype='float32')
                         for det_result in detection_results]
    
    return detection_results


def show_ann_from_json(coco_json: List[Dict[str, str]], image: np.ndarray, model: Dict[str, str]):
    model = _init_model(str(model['config']), str(model['checkpoint']))
    mmdet_result = coco_to_mmdet(coco_json)
    result_img = model.show_result(image, mmdet_result, show=False)
    return mmcv.bgr2rgb(result_img)
    

def main():
    WORK_DIR = Path(__file__).parents[1] / "mmdetection" / "checkpoints"
    MODELS = {
            "yolox_m": {
                "config": Path(WORK_DIR) / "yolox_m/bacteria_yolox_m_8x8_300e_coco.py",
                "checkpoint": Path(WORK_DIR) / "yolox_m/best_bbox_mAP_epoch_800.pth"
            }}
    img = cv2.imread('/home/badboy-002/github/senior_project/bacteria_img_jbing/20230109_TS008_1hr_1mindye_Kan_1x_25_R3D_D3D_CRC-1.tif')
    result = get_result(img, 
                        model_dir=str(MODELS['yolox_m']['config']),
                        check_point_dir=str(MODELS['yolox_m']['checkpoint']),
                        image_id=1
                        )
    img = show_ann_from_json(result, img, MODELS['yolox_m'])
    

def test_coco_to_csv(json_path: str):
    with open(json_path, 'r') as f:
        coco_json = json.load(f)
    coco_csv = coco_to_csv(coco_json, 1)
    return coco_csv

if __name__ == '__main__':
    main()
    


