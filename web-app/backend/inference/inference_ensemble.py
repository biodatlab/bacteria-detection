import sys
import os
from pathlib import Path
import argparse

sys.path.append(
    os.path.abspath(Path(__file__).parents[1])
    )

from typing import List, Any
import tifffile
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import time
import json
from glob import glob
from loguru import logger

import app
from app.colorize_model import colorize_image
from app.ensemble_model import ensemble_model
from app.extracting_features import extracting_features
from app.inference_model import coco_to_csv, get_result, show_ann_from_json


COCO_JSON_FORMAT = Path(__file__).parents[1] / "resource" / "coco_json_format.json"


def get_coco_json_template(coco_path: str):
    if not os.path.exists(coco_path):
        raise FileNotFoundError("File not found")

    with open(coco_path, "r") as coco_json:
        results = json.load(coco_json)
    return results


def read_image(imgs_folder: str):
    logger.debug(f'read file from {imgs_folder}')
    img_paths = glob(os.path.join(imgs_folder, '*.tif'))
    logger.debug(f'found {len(img_paths)} images')

    for img_path in img_paths:
        if img_path[-3:] != 'tif':
            # warning
            logger.warning('file {} is not tif file'.format(img_path))
        tiff_img = tifffile.imread(img_path)
        cv_img = cv2.imread(img_path)
        yield img_path, tiff_img, cv_img


def preprocess(img: np.ndarray, func_list: List[Any] = None):
    if img.shape != (3, 1024, 1024):
        raise ValueError('image shape is not 3, 1024, 1024')
    
    img = img.transpose(1, 2, 0)
    img = (cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))).astype(np.uint8)
    return img


def inference_ensemble_model(args_parser):

    imgs_folder = args_parser.imgs_folder

    time.sleep(1)
    app.clear_gpu()
    pred_response = {"image": [], "box_id": [], "json_results": {}, "features": None}
    result = get_coco_json_template(COCO_JSON_FORMAT)
    mask_model = tf.keras.models.load_model(app.MASK_MODEL_PATH)

    for idx, img_stack in enumerate(read_image(imgs_folder)):
        img_path, tif_img, cv_img = img_stack
        logger.debug('image path {}'.format(img_path))
        logger.debug('image shape {}'.format(tif_img.shape))
        # preprocess
        tif_img = preprocess(tif_img)
        original_img = tif_img.copy()

        # coloring image
        logger.debug('coloring image')
        color_img = (colorize_image(tif_img.copy()) * 255).astype(np.uint8)
        app.clear_gpu()

        # predict bbox
        logger.debug('predict bbox')
        coco_result = app.inference_bacteria_model(
            cv_img.copy(),
            idx
        )
        app.clear_gpu()

        # get image with bbox
        logger.debug('get image with bbox')
        img_with_bbox = show_ann_from_json(
            coco_result,
            color_img.copy(),
            app.MODELS['crcnn_r2101']
        )

        # feature extraction
        logger.debug('feature extraction')
        features, index_img = extracting_features(
            color_img.copy(),
            original_img.copy(),
            coco_result,
            mask_model
        )
        app.clear_gpu()

        logger.debug('Finish')
        break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference Ensemble Model",  
    )
    parser.add_argument(
        "--imgs_folder",
        type=str,
        help="Path to folder of images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_ensemble_model(args)