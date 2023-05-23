# Object detection of bacteria treated with antibiotics from microscopic images

Antibiotics are the primary drug for treating various kinds of infections occurring from bacteria and microbes.
They work mainly by blocking the vital pathway of those organisms and stopping them from multiplying. Previous
research shows that we can predict the antibiotics used on bacteria by visualizing its morphology. Here,
we present the object detection for detecting bacteria and identifying the antibiotics used on them.
There are 5 classes we are interested including bacteria treated with **Ampicillin**, **Ciprofloxacin**, **Rifampicin**,
**Tetracycline**, **Mecillinam**, **Kanamycin**, **Colistin**, and **Untreated**.

## Dataset

Current dataset contains **900 images**: Ampicillin (100), Ciprofloxacin (100), Rifampicin (100), Tetracycline (100), Mecillinam(100), Kanamycin(100), Colistin(100), 
and Untreated (200).

## Annotation tool

We annotated RGB version in PNG format of the bacteria images which is easier to visualize using [labelme](https://github.com/wkentaro/labelme).
Actual images are in TIFF format that we use for actual model training and testing.

## Results

Current result are as follows.\
Individual model
| Model                    | Backbone             | Head | Neck           | mAP   | mIOU  | AP (50) | AP(75) | AP (medium) | AP (large) |
|--------------------------|----------------------|------|----------------|-------|-------|---------|--------|-------------|------------|
| Cascade RCNN             | Res2Net-101 + DcnV2  | sabl | PAFPN + Dyhead | 0.652 | 0.800 | 0.808   | 0.762  | 0.677       | 0.692      |
| YOLOX_m                  | YOLOX_M + CSPDarknet | -    | PAFPN          | 0.621 | 0.755 | 0.902   | 0.835  | 0.711       | 0.796      |
| Cascade RCNN             | Res2Net-50 + DcnV2   | sabl | Dyhead         | 0.680 | 0.802 | 0.820   | 0.779  | 0.704       | 0.628      |
| Casade  RCNN + pre-train | Res2Net-50 + DcnV2   | sabl | PAFPN + Dyhead | 0.675 | 0.794 | 0.817   | 0.778  | 0.697       | 0.641      |

Weight box fusion model and mAP per class
| mIoU  |  mAP  | AP (50) | AP(75) | AP (medium) | AP (large) |
|-------|-------|---------|--------|-------------|------------|
| 0.753 | 0.699 | 0.863   | 0.796  | 0.717       | 0.675      |

|               |  unt  | Amp   | Cip   | Rif   | Tet   | Col   | Kan   | Mec   |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
| mAP per class | 0.406 | 0.839 | 0.897 | 0.727 | 0.574 | 0.794 | 0.517 | 0.837 |

Comparison with CellProfiler (traditional method)
|               | Bacteria detection model | CellProfiler | AP(75) |
|---------------|--------------------------|--------------|--------|
| Mean F1-score | 0.76                     | 0.863        | 0.796  |


## Installation 
```html
$ conda create -n your_env python=3.9.13 cython numpy
```
```
$ cd bacteria-detection
```
```
$ pip install -r requirements.txt
```

 ### Weight
Download required model [here](https://drive.google.com/drive/folders/1S8LEIkAcTxg5MJtzbsWkIeIt-Ayp5Mzz?usp=sharing)
 - move folder checkpoints to web-app/backend/mmdetection
 - move color_model_checkpoints and deepmac_1024x1024_coco17 to web-app/backend


## Members

- Korrawiz Chotayapa
- Thanyatorn Leethamchayo
- Piraya Chinnawong
- Titipat Achakulvisut
