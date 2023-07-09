# Load libraries
import cv2
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.color import rgb_colors
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define utilities

COLORS = [
    rgb_colors.cyan,
    rgb_colors.orange,
    rgb_colors.pink,
    rgb_colors.purple,
    rgb_colors.limegreen,
    rgb_colors.crimson,
] + [(color) for (name, color) in color.color_dict.items()]
random.shuffle(COLORS)

logging.disable(logging.WARNING)


def read_image(path):
    """Read an image and optionally resize it for better plotting."""
    with open(path, "rb") as f:
        img = Image.open(f)
        return np.array(img, dtype=np.uint8)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def create_detection_map(annotations):
    """Creates a dict mapping IDs to detections."""

    ann_map = {}
    for image in annotations["images"]:
        ann_map[image["id"]] = image["detections"]
    return ann_map


def get_mask_prediction_function(model):
    """Get single image mask prediction function using a model."""

    @tf.function
    def predict_masks(image, boxes):
        height, width, _ = image.shape.as_list()
        batch = image[tf.newaxis]
        boxes = boxes[tf.newaxis]

        detections = model(batch, boxes)
        masks = detections["detection_masks"]

        return reframe_box_masks_to_image_masks(masks[0], boxes[0], height, width)

    return predict_masks


def convert_boxes(boxes):
    xmin, ymin, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ymax = ymin + height
    xmax = xmin + width

    return np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)


# Copied from tensorflow/models
def reframe_box_masks_to_image_masks(
    box_masks, boxes, image_height, image_width, resize_method="bilinear"
):
    """Transforms the box masks back to full image masks.
    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.
    Args:
      box_masks: A tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
      resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
        'bilinear' is only respected if box_masks is a float.
    Returns:
      A tensor of size [num_masks, image_height, image_width] with the same dtype
      as `box_masks`.
    """
    resize_method = "nearest" if box_masks.dtype == tf.uint8 else resize_method

    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            denom = max_corner - min_corner
            # Prevent a divide by zero.
            denom = tf.math.maximum(denom, 1e-4)
            transformed_boxes = (boxes - min_corner) / denom
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1
        )
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

        # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
        # is static. This will help us run and test on TPUs.
        resized_crops = tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_indices=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            method=resize_method,
            extrapolation_value=0,
        )
        return tf.cast(resized_crops, box_masks.dtype)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype),
    )
    return tf.squeeze(image_masks, axis=3)


def plot_image_annotations(image, boxes, masks, darken_image=0.5):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_axis_off()
    image = (image * darken_image).astype(np.uint8)
    ax.imshow(image)

    height, width, _ = image.shape

    num_colors = len(COLORS)
    color_index = 0

    for box, mask in zip(boxes, masks):
        ymin, xmin, ymax, xmax = box
        ymin *= height
        ymax *= height
        xmin *= width
        xmax *= width

        color = COLORS[color_index]
        color = np.array(color)
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        mask = (mask > 0.5).astype(np.float32)
        color_image = np.ones_like(image) * color[np.newaxis, np.newaxis, :]
        color_and_mask = np.concatenate([color_image, mask[:, :, np.newaxis]], axis=2)

        # ax.imshow(color_and_mask, alpha=0.5)

        color_index = (color_index + 1) % num_colors
    # plt.show()

    return ax


def preprocess_bbox(bboxes: List[int]) -> np.ndarray:
    return convert_boxes(np.array(bboxes)) / 1024


# loading model and init prediction function


def predict_mask(
    color_image: np.ndarray, anns: List[Dict[str, Any]], model: Any
) -> np.ndarray:
    """
    Predict mask from color image and bounding boxes
    :param color_image: color image
    :param anns: list of annotations
    :param model: model
    :return: mask
    """
    prediction_function = get_mask_prediction_function(model)
    boxes = [ann["bbox"] for ann in anns]
    masks = prediction_function(
        tf.convert_to_tensor(color_image),
        tf.convert_to_tensor(preprocess_bbox(boxes), dtype=tf.float32),
    )
    plot_image_annotations(
        color_image, preprocess_bbox(boxes), masks.numpy(), darken_image=0.75
    )
    return masks.numpy()


def get_membrane_features(masks: np.ndarray) -> pd.DataFrame:
    total_contours = 0
    contour_image = np.zeros_like(masks[0])

    perimeter_list = []
    area_list = []
    length_list = []
    width_list = []
    mask_order_list = []
    # for bounding box position
    x_list = []
    y_list = []
    w_list = []
    h_list = []

    for i, mask in enumerate(masks):
        gray = (mask * 255).astype(np.uint8)
        ret, thresh = cv.threshold(gray, 100, 255, 0)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        for j, cnt in enumerate(contours):
            # perimeter
            perimeter = cv.arcLength(cnt, False)
            perimeter_list.append(perimeter)
            # area
            area = cv.contourArea(cnt)
            area_list.append(area)
            # length width
            rect = cv.minAreaRect(cnt)
            width = min(rect[1])
            length = max(rect[1])
            length_list.append(length)
            width_list.append(width)
            mask_order = total_contours + j + 1
            mask_order_list.append(mask_order)
            # color = tuple(np.random.randint(0, 255, 3).tolist())
            color = (255, 0, 0)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(contour_image, (x, y), (x + w, y + h), color, 2)
            x_list.append(x)
            y_list.append(y)
            w_list.append(w)
            h_list.append(h)
            text_pos = tuple(cnt[0][0] + np.array([0, -8]))
            text = f"{mask_order}"
            cv.putText(
                contour_image, text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )

        # Update the total count of contours found so far
        total_contours += len(contours)

    boxes_membrane = [
        [x, y, w, h] for x, y, w, h in zip(x_list, y_list, w_list, h_list)
    ]

    membrane_df = pd.DataFrame(
        {
            "Membrane ID": mask_order_list,
            "Membrane Perimeter": perimeter_list,
            "Membrane Area": area_list,
            "Membrane Length": length_list,
            "Membrane Width": width_list,
        }
    )

    return membrane_df, boxes_membrane, contour_image


def get_membrane_area(color_img: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
    membrane_only = color_img[..., 0].copy()
    kernel = np.ones((5, 5), np.uint8)
    area_list = []
    for i, mask in enumerate(masks):
        croped_membrane = membrane_only * (mask > 0.5)
        _, thresh = cv.threshold(croped_membrane, 50, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = sum(list(map(cv2.contourArea, contours)))

        area_list.append({"Membrane ID": i + 1, "Membrane Area": area})

    membrane_area_df = pd.DataFrame(area_list)
    membrane_area_df.to_csv("membrane_area.csv", index=False)

    return membrane_area_df


def get_dna_features(color_img: np.ndarray):
    dna_image = color_img.copy()
    blue_channel = dna_image[:, :, 2]
    ret, thresh = cv.threshold(blue_channel, 100, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    dna_perimeter_list = []
    dna_area_list = []
    dna_width_list = []
    dna_object_order = []
    dna_length_list = []
    x_list = []
    y_list = []
    w_list = []
    h_list = []
    for i, cnt in enumerate(contours):
        perimeter = cv.arcLength(cnt, False)
        area = cv.contourArea(cnt)
        rect = cv.minAreaRect(cnt)
        width = min(rect[1])
        length = max(rect[1])
        dna_perimeter_list.append(perimeter)
        dna_area_list.append(area)
        dna_width_list.append(width)
        dna_length_list.append(length)
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(dna_image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        x_list.append(x)
        y_list.append(y)
        h_list.append(h)
        w_list.append(w)
        object_order = i + 1
        dna_object_order.append(object_order)
        text_pos = tuple(cnt[0][0] + np.array([5, -5]))
        text = cv.putText(
            dna_image,
            f"{object_order}",
            text_pos,
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            1,
        )

    dna_df = pd.DataFrame(
        {
            "DNA ID": dna_object_order,
            "DNA Perimeter": dna_perimeter_list,
            "DNA Area": dna_area_list,
            "DNA Width": dna_width_list,
            "DNA Length": dna_length_list,
        }
    )
    boxes_dna_list = [
        [x, y, w, h] for x, y, w, h in zip(x_list, y_list, w_list, h_list)
    ]
    return dna_df, boxes_dna_list, dna_image


def get_position_from_component_boxes(
    boxes_membrane: List[int], boxes_dna: List[int]
) -> pd.DataFrame:
    boxes_results = []

    for i, membrane_box in enumerate(boxes_membrane):
        for j, dna_box in enumerate(boxes_dna):
            x1, y1, w1, h1 = dna_box
            x2, y2, w2, h2 = membrane_box
            if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                boxes_results.append((j + 1, dna_box, i + 1, membrane_box))
    return pd.DataFrame(
        boxes_results,
        columns=["DNA ID", "DNA Position", "Membrane ID", "Membrane Position"],
    )


def get_intensity_features(
    original_img: np.ndarray, features_df: pd.DataFrame
) -> pd.DataFrame:
    intsty_img = original_img
    print(f"image dtype: {intsty_img.dtype}")
    # convert to float:
    intensity_features = []
    membrane_pos = features_df[["Membrane ID", "Membrane Position"]]
    for ids, box in zip(
        membrane_pos["Membrane ID"].tolist(), membrane_pos["Membrane Position"].tolist()
    ):
        if not isinstance(box, list):
            intensity_feat = {
                "Membrane ID": ids,
                "Green Max Intensity": 0,
                "Green Min Intensity": 0,
                "Green Mean Intensity": 0,
                "Green Median Intensity": 0,
                "Green STD Intensity": 0,
                "Blue Max Intensity": 0,
                "Blue Min Intensity": 0,
                "Blue Mean Intensity": 0,
                "Blue Median Intensity": 0,
                "Blue STD Intensity": 0,
            }
            intensity_features.append(intensity_feat)
            continue

        roi = intsty_img[
            int(box[0]) : int(box[0] + box[2]), int(box[1]) : int(box[1] + box[3]), :
        ]

        intensity_feat = {
            "Membrane ID": ids,
            "Green Max Intensity": np.amax(roi[..., 1]),
            "Green Min Intensity": np.amin(roi[..., 1]),
            "Green Mean Intensity": np.mean(roi[..., 1]),
            "Green Median Intensity": np.median(roi[..., 1]),
            "Green STD Intensity": np.std(roi[..., 1]),
            "Blue Max Intensity": np.amax(roi[..., 2]),
            "Blue Min Intensity": np.amin(roi[..., 2]),
            "Blue Mean Intensity": np.mean(roi[..., 2]),
            "Blue Median Intensity": np.median(roi[..., 2]),
            "Blue STD Intensity": np.std(roi[..., 2]),
        }
        intensity_features.append(intensity_feat)

    intensity_features = pd.DataFrame(intensity_features)
    features_df = pd.merge(features_df, intensity_features, on="Membrane ID")
    return features_df


def get_features_from_df(
    membrane_df: pd.DataFrame,
    dna_df: pd.DataFrame,
    boxes_df: pd.DataFrame,
    original_img: np.ndarray,
) -> pd.DataFrame:
    dna_counts = boxes_df["Membrane ID"].value_counts().sort_index()
    dna_counts_df = pd.DataFrame(
        {"Membrane ID": dna_counts.index, "No. of nucleoids": dna_counts.values}
    )
    dna_merged_df = pd.merge(boxes_df, dna_df, on="DNA ID")
    dna_area = dna_merged_df.groupby("Membrane ID")["DNA Area"].idxmax()
    dna_result_df = dna_merged_df.loc[
        dna_area,
        [
            "Membrane ID",
            "DNA Perimeter",
            "DNA Area",
            "DNA Width",
            "DNA Length",
            "Membrane Position",
        ],
    ]
    final_results_df = membrane_df.merge(
        dna_result_df, on="Membrane ID", how="left"
    ).merge(dna_counts_df, on="Membrane ID", how="left")
    final_results_df = get_intensity_features(original_img, final_results_df)
    return final_results_df


def labeling_img_box(color_img: np.ndarray, bbox_img: np.ndarray) -> np.ndarray:
    # overlay the image with the bounding box
    color_img[bbox_img != 0] = [255, 255, 255]

    return color_img


def extracting_features(
    color_img: np.ndarray,
    original_img: np.ndarray,
    annotations: List[Dict[str, Any]],
    model: Any,
) -> pd.DataFrame:
    masks = predict_mask(color_img.copy(), annotations, model)
    membrane_df, boxes_membrane, contour_image = get_membrane_features(masks)
    # boxes = [ann['bbox'] for ann in annotations]
    membrane_new_area_df = get_membrane_area(color_img, masks)
    membrane_df = membrane_df.drop(["Membrane Area"], axis=1)
    membrane_df = pd.merge(membrane_new_area_df, membrane_df, on="Membrane ID")
    contour_image = contour_image.astype("uint8")
    result_img = labeling_img_box(color_img.copy(), contour_image)
    dna_df, boxes_dna, dna_image = get_dna_features(color_img.copy())
    boxes_df = get_position_from_component_boxes(boxes_membrane, boxes_dna)
    final_results_df = get_features_from_df(membrane_df, dna_df, boxes_df, original_img)
    print("final result shape: ", final_results_df.shape)
    # plt.imshow(result_img)
    # plt.show()
    return final_results_df, result_img


def testing_get_mask():
    model = tf.keras.models.load_model(
        "/home/badboy-002/github/senior_project/web-app/deepmac_1024x1024_coco17/saved_model"
    )
    demo_folder = Path(__file__).parents[1] / "demo_files"
    demo_img = Path(demo_folder) / "color_demo_img.png"
    demo_img = cv.imread(str(demo_img))
    demo_img = cv.cvtColor(demo_img, cv.COLOR_BGR2RGB)

    ori_img = Path(demo_folder) / "original_demo_img.png"
    ori_img = cv.imread(str(ori_img))
    ori_img = cv.cvtColor(ori_img, cv.COLOR_BGR2RGB)

    with open(str(Path(demo_folder) / "result_demo.json")) as f:
        annotations = json.load(f)
    result, _ = extracting_features(demo_img, ori_img, annotations, model)
    result.to_csv("test_result.csv")


if __name__ == "__main__":
    testing_get_mask()
