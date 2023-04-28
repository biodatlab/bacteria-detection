from typing import Optional
from fastapi import FastAPI, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, Dict, List, Any
import numpy as np
from tempfile import NamedTemporaryFile
from pathlib import Path, PosixPath
import shutil
import cv2
import os
import mmcv, json
import asyncio
from fastapi.responses import FileResponse, StreamingResponse, Response
import io
import os.path as op
from os import listdir, remove
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from inference_model import get_result, show_ann_from_json, coco_to_csv
from ensemble_model import ensemble_model
from loguru import logger
import pylab
import gc
import torch
from pycocotools import coco
import PIL.Image as Image
import base64
from colorize_model import colorize_image
import matplotlib.pyplot as plt
import io
import skimage
import tifffile
import pandas as pd
import yaml
import tensorflow as tf
import gc
import threading
from extracting_features import extracting_features


# /home/badboy-002/github/senior_project/bacteria-detection/mmdetection/work_dirs
PRJ_ROOT = Path(__file__).parents[2] / 'web-app'
CONFIG_DIR = PRJ_ROOT / 'backend' / 'config.yaml'
DEMO_IMG_PATH = PRJ_ROOT / 'backend' / 'demo_image'
DEMO_FILES_PATH = PRJ_ROOT / 'backend' / 'demo_files'
WORK_DIR = Path('mmdetection') / 'checkpoints'
SCORE_THRESHOLD = 0.3


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open(CONFIG_DIR, 'r') as f:
    config = yaml.safe_load(f)

MODELS = config["detection_model"]
MASK_MODEL_PATH = str(PRJ_ROOT / config['segmentation_model'])
for model in MODELS.keys():
    for key, value in MODELS[model].items():
        MODELS[model][key] = str(Path(PRJ_ROOT) / value)

def thresholding_box(anns_list: List[Dict[str, Any]], threshold=0.3):
    return [ann for ann in anns_list if ann['score'] > threshold]



def inference_bacteria_model(image: np.ndarray,
                             image_id: int,
                             model_dict=MODELS):
    global SCORE_THRESHOLD
    # get result from each model
    results = []
    for model in model_dict.keys():
        results.append(get_result(image, 
                   str(model_dict[model]["config"]),
                   str(model_dict[model]["checkpoint"]),
                   image_id)
                       )
    # get number of image in each model and check if they are the same
    num_image = []
    for result in results:
        num_image.append(len(set([i["image_id"] for i in result])))
        
    if len(set(num_image)) > 1:
        logger.warning("Number of image in each model is not the same the using the least number of image.")
        num_image = min(num_image)
        
    elif len(set(num_image)) == 1:
        num_image = num_image[0]
        
    else:
        return {"error": "No image found"}
    
    # ensemble model
    ensemble_result =  ensemble_model(results, num_image, image_id)
    return thresholding_box(ensemble_result, threshold=SCORE_THRESHOLD)

def clear_gpu():
    gc.collect()

def get_features():
    features_bac = {
        'membrane_width': 10,
        'membrane_high': 10,
        'membrane_permeter': 10,
        'membrane_area': 10,
    }
    
    return features_bac


def dict_to_csv(json_dict):
    df = pd.DataFrame(json_dict, index=[0])
    return df.to_csv(index=False)
    



def array_to_byte(image: np.array) -> str:
    buffer = cv2.imencode('.png', image)[1]
    bytes_array = np.array(buffer).tobytes()
    return base64.b64encode(bytes_array).decode('utf-8')



@app.get("/")
async def index() -> None:
    """ This is the index page """
    return {
        "name": "MMdet API",
    }


@app.post('/detect/')
async def create_file(answer_images: List[UploadFile]):
    # global SCORE_THRESHOLD
    # print('score threshold:', SCORE_THRESHOLD)
    import time
    time.sleep(1)

    clear_gpu()
    pred_response = {
        "image": [],
        "box_id": [],
        "json_results": {},
        "features": None
    }
    with open('./resource/coco_json_format.json', 'r') as coco_json:
        results = json.load(coco_json)
    
    mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)

    for idx, answer_image in enumerate(answer_images):

        img_input_byte = bytearray(await answer_image.read())
        img = tifffile.imread(io.BytesIO(img_input_byte.copy()))
        img = img.transpose(1, 2, 0)
        original_img = img
        print("image input dtype", img.dtype)
        img = (cv2.convertScaleAbs(img, alpha=(255.0/65535.0))).astype(np.uint8)
        cv2.imwrite(os.path.join(str(DEMO_FILES_PATH), f"original_demo_img.png"),
                    cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
    
        color_img = (colorize_image(img.copy())*255).astype(np.uint8)
        cv2.imwrite(os.path.join(str(DEMO_FILES_PATH), f"color_demo_img.png"),
                    cv2.cvtColor(color_img.copy(), cv2.COLOR_RGB2BGR))
        clear_gpu()
        results['images'].append({
            "height": img.shape[0],
            "width": img.shape[1],
            "id": idx,
            "file_name": str(answer_image.filename)
        })
        # read image from answer image

        img_cv = np.asarray(img_input_byte.copy(), dtype="uint8")
        img_cv = cv2.imdecode(img_cv, cv2.IMREAD_COLOR)
        coco_result = inference_bacteria_model(img_cv.copy(), idx)
        with open(os.path.join(str(DEMO_FILES_PATH),f"result_demo.json"), 'w') as f:
            json.dump(coco_result, f)
        
        clear_gpu()
        results['annotations'] += coco_result
        # ได้รูปตรงนี้
        ann_img = show_ann_from_json(coco_result, color_img.copy(), MODELS["crcnn_r2101"])
        clear_gpu()
        cv2.imwrite(os.path.join(str(DEMO_FILES_PATH),"output.png"), ann_img.copy())
        features, index_img = extracting_features(color_img.copy(),
                                                  original_img.copy(),
                                                  coco_result,
                                                  mask_model)
        features['file_name'] = [str(Path(str(answer_image.filename)).stem) 
                                 for _ in range(len(features))]

        if not isinstance(pred_response['features'], pd.DataFrame):
            pred_response['features'] = features
        else:
            pred_response['features'] = pd.concat([
                pred_response['features'],
                features,
            ], axis=0)
        logger.info(pred_response['features'].shape)
        cv2.imwrite(os.path.join(str(DEMO_FILES_PATH),"idx_img.png"),
                    cv2.cvtColor(index_img.copy(), cv2.COLOR_RGB2BGR))
        bytes_string = array_to_byte(ann_img)
        index_img_byte = array_to_byte(cv2.cvtColor(index_img, cv2.COLOR_RGB2BGR))

        # append to dictionary
        pred_response['image'].append(bytes_string)
        pred_response['box_id'].append(index_img_byte)
        time.sleep(5)

    pred_response['json_results'] = coco_to_csv(results)
    # save as csv
    with open(os.path.join(str(DEMO_FILES_PATH),'results.csv'), 'w') as f:
        f.write(coco_to_csv(results))
        
    pred_response['features'] =  pred_response['features'].to_csv(
        index=False
    )
    with open(os.path.join(str(DEMO_FILES_PATH),'features123.csv'), 'w') as f:
        f.write(pred_response['features'])
    print("type:", type(pred_response['features']))
    with open('pred_response.json', 'w') as f:
        json.dump(pred_response, f)
    print("finish")
    return pred_response


@app.put('/threshold/')
async def update_threshold(threshold: float):
    global SCORE_THRESHOLD
    SCORE_THRESHOLD = threshold
    logger.info(f'score threshold:{SCORE_THRESHOLD}')
    return {"threshold": SCORE_THRESHOLD}


if __name__ == "__main__":
    test_img_path = '/home/badboy-002/github/senior_project/bacteria_img_jbing/20221228_TS008_1hr_1mindye_KAN_1x_47_R3D_D3D_CRC-1.tif'