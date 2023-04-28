import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import skimage

import os
from pathlib import Path
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
tf.config.run_functions_eagerly(True)
from tensorflow import keras
import segmentation_models as sm
from glob import glob
from tqdm import tqdm
import yaml

PRJ_ROOT = Path(__file__).parents[2] / 'web-app'
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
    # mask[...,2] = cv2.morphologyEx(mask[...,2], cv2.MORPH_OPEN, kernel, iterations = 1)
    mask[...,2] = cv2.morphologyEx(mask[...,2], cv2.MORPH_CLOSE, kernel, iterations = 3)
    
    
    return mask

def colorize_image(image: np.array) -> np.array:
    
    model = init_segmentatio_model()
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).squeeze()
    
    # plt.figure(figsize=(10,15))
    # plt.subplot(121); plt.imshow(pr_mask)
    # plt.subplot(122); plt.imshow(post_process_mask(pr_mask))
    # plt.show()
    return post_process_mask(pr_mask)


if __name__ == "__main__":
    img = '/home/badboy-002/github/senior_project/bacteria_img_jbing/20221007_TS008_1hr_CIP_5x_05_R3D_D3D_CRC-1.tif'
    image = skimage.io.imread(img)
    image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
    pr_mask = colorize_image(image)
    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(
        '/home/badboy-002/github/senior_project/web-app/backend/demo_files',
        str(Path(img).stem) + '.png'
    ), pr_mask*255)

    plt.imshow(pr_mask)
    plt.show()

    # img_test_path = '/home/badboy-002/github/senior_project/web-app/backend/models/unet/data/images_test/images'
    # img_tests = glob(img_test_path + '/**/*.tif', recursive=True)
    # print(img_tests[0])
    # for img in tqdm(img_tests):
    #     image = skimage.io.imread(img)
    #     image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
    #     pr_mask = colorize_image(image)
    #     pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(os.path.join(
    #         '/home/badboy-002/github/senior_project/web-app/backend/models/unet/data/images_test', 'output',
    #         str(Path(img).stem) + '.png'
    #     ), pr_mask*255)