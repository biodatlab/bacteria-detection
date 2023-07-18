import base64
import gc
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Annotated

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile
import yaml
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .colorize_model import colorize_image
from .ensemble_model import ensemble_model
from .extracting_features import extracting_features
from .inference_model import coco_to_csv, get_result, show_ann_from_json

# /home/badboy-002/github/senior_project/bacteria-detection/mmdetection/work_dirs
PRJ_ROOT = Path(__file__).parents[3] / "web-app"
CONFIG_DIR = PRJ_ROOT / "backend" / "config.yaml"
DEMO_IMG_PATH = PRJ_ROOT / "backend" / "demo_image"
DEMO_FILES_PATH = PRJ_ROOT / "backend" / "demo_files"
SCORE_THRESHOLD = 0.3
COCO_TEMPLATE = PRJ_ROOT / "backend" / "resource" / "coco_json_format.json"


gpus = tf.config.experimental.list_physical_devices("GPU")
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


with open(CONFIG_DIR, "r") as f:
    config = yaml.safe_load(f)


MODELS = config["detection_model"]
MASK_MODEL_PATH = str(PRJ_ROOT / config["segmentation_model"])
for model in MODELS.keys():
    for key, value in MODELS[model].items():
        MODELS[model][key] = str(Path(PRJ_ROOT) / value)


def thresholding_box(anns_list: List[Dict[str, Any]], threshold=0.3):
    return [ann for ann in anns_list if ann["score"] > threshold]


def inference_bacteria_model(image: np.ndarray, image_id: int, model_dict=MODELS):
    global SCORE_THRESHOLD
    # get result from each model
    results = []
    logger.info(f"Using confidence score >: {SCORE_THRESHOLD}")
    for model in model_dict.keys():
        results.append(
            get_result(
                image,
                str(model_dict[model]["config"]),
                str(model_dict[model]["checkpoint"]),
                image_id,
            )
        )
    # get number of image in each model and check if they are the same
    num_image = []
    for result in results:
        num_image.append(len(set([i["image_id"] for i in result])))

    if len(set(num_image)) > 1:
        logger.warning(
            "Number of image in each model is not the same the using the least number of image."
        )
        num_image = min(num_image)

    elif len(set(num_image)) == 1:
        num_image = num_image[0]

    else:
        return {"error": "No image found"}
    # ensemble model
    ensemble_result = ensemble_model(results, num_image, image_id)
    return thresholding_box(ensemble_result, threshold=SCORE_THRESHOLD)


def clear_gpu():
    gc.collect()


def dict_to_csv(json_dict):
    df = pd.DataFrame(json_dict, index=[0])
    return df.to_csv(index=False)


def array_to_byte(image: np.array) -> str:
    buffer = cv2.imencode(".png", image)[1]
    bytes_array = np.array(buffer).tobytes()
    return base64.b64encode(bytes_array).decode("utf-8")


@app.get("/")
async def index() -> None:
    """This is the index page"""
    return {
        "name": "MMdet API",
    }


@app.post("/detect/")
async def create_file(answer_images: List[UploadFile]):

    time.sleep(1)
    clear_gpu()

    pred_response = {"image": [], "box_id": [], "json_results": {}, "features": None}
    with open(COCO_TEMPLATE, "r") as coco_json:
        results = json.load(coco_json)

    logger.info("Start inference model")
    logger.info(f"loading model...")
    mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)
    logger.info(f"model loaded!")

    for idx, answer_image in enumerate(answer_images):

        # read image from answer image
        img_input_byte = bytearray(await answer_image.read())
        img = tifffile.imread(io.BytesIO(img_input_byte.copy()))

        # preprocessing image
        img = img.transpose(1, 2, 0)
        original_img = img
        img = (cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))).astype(np.uint8)
        cv2.imwrite(
            os.path.join(str(DEMO_FILES_PATH), f"original_demo_img.png"),
            cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR),
        )
        # add image to final response
        results["images"].append(
            {
                "height": img.shape[0],
                "width": img.shape[1],
                "id": idx,
                "file_name": str(answer_image.filename),
            }
        )
        # colorize image
        color_img = (colorize_image(img.copy()) * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(str(DEMO_FILES_PATH), f"color_demo_img.png"),
            cv2.cvtColor(color_img.copy(), cv2.COLOR_RGB2BGR),
        )
        clear_gpu()
        
        # read image from answer image in cv2 format
        img_cv = np.asarray(img_input_byte.copy(), dtype="uint8")
        img_cv = cv2.imdecode(img_cv, cv2.IMREAD_COLOR)
        coco_result = inference_bacteria_model(img_cv.copy(), idx)

        # saving result to json
        with open(os.path.join(str(DEMO_FILES_PATH), f"result_demo.json"), "w") as f:
            json.dump(coco_result, f)
        clear_gpu()

        # add annotation to final response
        results["annotations"] += coco_result
        ann_img = show_ann_from_json(
            coco_result, color_img.copy(), MODELS["crcnn_r2101"]
        )
        # saving annotation image
        cv2.imwrite(os.path.join(str(DEMO_FILES_PATH), "output.png"), ann_img.copy())

        # extracting features
        features, index_img = extracting_features(
            color_img.copy(), original_img.copy(), coco_result, mask_model
        )
        features["file_name"] = [
            str(Path(str(answer_image.filename)).stem) for _ in range(len(features))
        ]

        # adding features to final response
        if not isinstance(pred_response["features"], pd.DataFrame):
            pred_response["features"] = features
        else:
            pred_response["features"] = pd.concat(
                [
                    pred_response["features"],
                    features,
                ],
                axis=0,
            )

        # saving image with index
        cv2.imwrite(
            os.path.join(str(DEMO_FILES_PATH), "idx_img.png"),
            cv2.cvtColor(index_img.copy(), cv2.COLOR_RGB2BGR),
        )

        # convert image to byte for Frontend
        bytes_string = array_to_byte(ann_img)
        index_img_byte = array_to_byte(cv2.cvtColor(index_img, cv2.COLOR_RGB2BGR))

        # append to dictionary
        pred_response["image"].append(bytes_string)
        pred_response["box_id"].append(index_img_byte)

    # save annotation result to csv format in final response
    pred_response["json_results"] = coco_to_csv(results)

    # save annotation result to csv format in local
    with open(os.path.join(str(DEMO_FILES_PATH), "results.csv"), "w") as f:
        f.write(coco_to_csv(results))

    # change features to csv format in final response
    pred_response["features"] = pred_response["features"].to_csv(index=False)

    # save features to csv format in local
    with open(os.path.join(str(DEMO_FILES_PATH), "features.csv"), "w") as f:
        f.write(pred_response["features"])

    # save final response to json format in local
    with open("pred_response.json", "w") as f:
        json.dump(pred_response, f)

    logger.info("Finish")
    return pred_response


@app.put("/threshold/")
async def update_threshold(threshold: str):
    threshold = float(threshold)
    global SCORE_THRESHOLD
    logger.info(f"input threshold:{type(threshold)} {threshold}")
    SCORE_THRESHOLD = threshold
    print("score threshold:", SCORE_THRESHOLD)
    logger.info(f"update score threshold to:{SCORE_THRESHOLD}")
    return {"threshold": SCORE_THRESHOLD}

