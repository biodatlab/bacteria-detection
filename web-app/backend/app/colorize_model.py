import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
tf.config.run_functions_eagerly(True)
from glob import glob

import segmentation_models as sm
import yaml

PRJ_ROOT = Path(__file__).parents[3] / 'web-app'
CONFIG_DIR = PRJ_ROOT / 'backend/config.yaml'


with open(CONFIG_DIR, 'r') as f:
    config = yaml.safe_load(f)
    color_model_path = os.path.join(PRJ_ROOT, config["colorize_model"]["checkpoint"])


def init_segmentatio_model():
    # load your trained model
    BACKBONE = 'resnet101'
    CLASSES = ['membrane', 'dna']
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
    model.load_weights(color_model_path)
    return model


def post_process_mask(mask: np.array) -> np.array:
    mask[...,1] = mask[...,1] * 0
    # morphology closing
    kernel = np.ones((3,3),np.uint8)
    mask[...,2] = cv2.morphologyEx(mask[...,2], cv2.MORPH_CLOSE, kernel, iterations = 3)
    return mask


def colorize_image(image: np.array) -> np.array:
    model = init_segmentatio_model()
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).squeeze()
    return post_process_mask(pr_mask)


if __name__ == "__main__":

    test_images = glob(os.path.join(PRJ_ROOT, 'backend/resource/images_test/images/*.tif'))
    test_results_path = os.path.join(PRJ_ROOT, 'backend/resource/images_test/results')
    print(test_images)

    import tifffile
    img_path = os.path.join(PRJ_ROOT,'backend/resource/images_test/images/20221003_TS008_1hr_1mindye_TET_5x_05_R3D_D3D_CRC-1.tif')
    img = tifffile.imread(img_path)
    img = img.transpose(1, 2, 0)
    original_img = img
    print("image input dtype", img.dtype)
    img = (cv2.convertScaleAbs(img, alpha=(255.0/65535.0)))
    img = img.astype(np.uint8)
    plt.imshow(img*20); plt.show()
    color_img = (colorize_image(img.copy()*20)*255).astype(np.uint8)
    plt.imshow(color_img); plt.show()