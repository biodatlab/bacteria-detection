from itertools import groupby
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ensemble_boxes import *
from typing import List, Dict, Any, Tuple
from pycocotools.coco import COCO
import numpy as np
from .inference_model import get_result

CocoJson = List[Dict[str, Any]]
WIDTH = 1024
HEIGHT = 1024


def coco_to_pascal_voc(box, WIDTH, HEIGHT):
    x1, y1, w, h = box
    return [x1 / WIDTH, y1 / HEIGHT, (x1 + w) / WIDTH, (y1 + h) / HEIGHT]


# Convert Yolo bb to Pascal_voc bb
def yolo_to_pascal_voc(box, WIDTH, HEIGHT):
    image_w = WIDTH
    image_h = HEIGHT
    (
        x_center,
        y_center,
        w,
        h,
    ) = box
    x1 = ((2 * x_center) - w) / 2
    y1 = ((2 * y_center) - h) / 2
    x2 = x1 + w
    y2 = y1 + h
    return [x1 / WIDTH, y1 / HEIGHT, x2 / WIDTH, y2 / HEIGHT]


def extract_bbox(result, WIDTH, HEIGHT):
    groupby(result, key=lambda x: x["image_id"])
    list_result = [list(v) for k, v in groupby(result, key=lambda x: x["image_id"])]
    bbox_list = []
    score_list = []
    label_list = []

    for im in list_result:
        bbox = [
            list(
                map(
                    lambda x: round(x, 3),
                    coco_to_pascal_voc(box["bbox"], WIDTH, HEIGHT),
                )
            )
            for box in im
        ]

        score = [scr["score"] for scr in im]
        label = [lab["category_id"] for lab in im]
        bbox_list.append(bbox)
        score_list.append(score)
        label_list.append(label)

    dict_det = dict(boxes=bbox_list, scores=score_list, labels=label_list)
    return dict_det


def read_pred_result(results_dir: List[str]) -> Tuple[CocoJson, int]:
    if not isinstance(results_dir, list):
        raise TypeError("results_dir must be a list")

    result_coco_list = []
    num_img = []
    for result in results_dir:
        with open(result, "r") as coco_json:
            annos = json.load(coco_json)
            result_coco_list.append(annos)
            coco_json.close()
            num_img.append(len(set([i["image_id"] for i in annos])))
    
    if len(set(num_img)) != 1:
        raise ValueError("The number of images in each result must be the same")
    
    num_images = num_img[0]
    return result_coco_list, num_images


def get_pool_bbox(results_list: List[CocoJson]) -> List[CocoJson]:
    return [extract_bbox(result, WIDTH, HEIGHT) for result in results_list]


def add_bbox(
    image: np.ndarray, boxes: List[List[int]], color: Tuple[int, int, int] = (255, 0, 0)
):
    image_copy = image.copy()
    for box in boxes:
        try:
            image_copy = cv2.rectangle(
                image_copy,
                (int(box[0] * WIDTH), int(box[1] * HEIGHT)),
                (int((box[2]) * WIDTH), int((box[3]) * HEIGHT)),
                color,
                2,
            )
        except:
            continue

    return image_copy


def ensemble_bbox(
    img_id: int,
    model_results: List[CocoJson],
    weights: List[float] = [3.0, 2.0, 1.0],
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0001,
    sigma: float = 0.1,
    mode: str = "wbf",
) -> CocoJson:
    boxes_list = []
    scores_list = []
    labels_list = []
    for result in model_results:
        boxes_list.append(result["boxes"][img_id])
        scores_list.append(result["scores"][img_id])
        labels_list.append(result["labels"][img_id])

    if mode == "wbf":
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
    elif mode == "nms":
        boxes, scores, labels = nms(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr
        )
    elif mode == "snms":
        boxes, scores, labels = soft_nms(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            thresh=skip_box_thr,
        )
    return dict(boxes=boxes, scores=scores, labels=labels)

def denormalize_bbox(pascal_bbox: List[float], WIDTH: int=1024,HEIGHT: int=1024) -> List[float]:
    xmin, ymin, xmax, ymax = pascal_bbox
    # check if all box is NaN
    if np.isnan(xmin) and np.isnan(ymin) and np.isnan(xmax) and np.isnan(ymax):
        xmin, ymin, xmax, ymax = [0]*4
    
    xmin = round(xmin*WIDTH,4)
    ymin = round(ymin*HEIGHT,4)
    xmax = round(xmax*WIDTH,4)
    ymax = round(ymax*HEIGHT,4)
    coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
    coco_bbox = [int(i) for i in coco_bbox]
    return coco_bbox


def ensemble_model(results_list: List[CocoJson], num_image, idx) -> CocoJson:
    ensemble_box_coco = []
    # results_list, num_image = read_pred_result(model_results)
    model_boxes = get_pool_bbox(results_list)
    for num in range(num_image):
        ensemble_wbf = ensemble_bbox(num, model_boxes, weights = [1,1,1], iou_thr = 0.8, skip_box_thr = 0.001,  sigma = 0.5, mode = 'wbf')

        #iterate throgh each box in image
        for i in range( len(ensemble_wbf['boxes']) ):

            anno_dict =  {
                "id" : i,
                "image_id": idx,
                "bbox": denormalize_bbox((ensemble_wbf['boxes'][i]).tolist()),
                "score": ensemble_wbf['scores'][i].round(4),
                "category_id": int(ensemble_wbf['labels'][i])
            }
            ensemble_box_coco.append(anno_dict)
            
            
    return ensemble_box_coco

def main():
    import skimage, yaml
    img = cv2.imread('/home/badboy-002/github/senior_project/bacteria_img_jbing/20221228_TS008_1hr_1mindye_KAN_1x_22_R3D_D3D_CRC-1.tif'
)
    img = (cv2.convertScaleAbs(img, alpha=(255.0/65535.0))).astype(np.uint8)
    result = get_result(img, 
                        '/home/badboy-002/github/senior_project/bacteria-detection2/mmdetection/work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/bacteria_r2101dcn_dyhead.py', 
                        '/home/badboy-002/github/senior_project/bacteria-detection2/mmdetection/work_dirs/crcnn_r2101_dcnv2_dyhead_15-2-23/best_bbox_mAP_epoch_16.pth',
                        1)
    num_image = len(set([i["image_id"] for i in result]))
    ensemble_results = ensemble_model([result, result, result], num_image)
    # dump to json
    with open('ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f)

if __name__ == "__main__":
    main()