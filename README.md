# Automatic detection and morphological feature extraction of E.Coli under antibiotic treatments

Antibiotics are the primary drug for treating various kinds of infections occurring from bacteria and microbes.
They work mainly by blocking the vital pathway of those organisms and stopping them from multiplying. Previous
research shows that we can predict the antibiotics used on bacteria by visualizing their morphology. Here,
we present object detection for detecting bacteria and identifying the antibiotics used on them with their mophological features for example DNA intensity, contour area, and min areaRect.

There are 8 classes we are interested including bacteria treated with **Ampicillin**, **Ciprofloxacin**, **Rifampicin**,
**Tetracycline**, **Mecillinam**, **Kanamycin**, **Colistin**, and **Untreated**.

<img src="./Bacteria images/readme_image/diagram.jpg"/>

## Object detection

The current results of single models and ensemble models are as follows.


| Model                    | Backbone             | Head | Neck           | mAP   | mIOU  | AP(50) | AP(75) | AP (medium) | AP (large) | Checkpoint|
|--------------------------|----------------------|------|----------------|-------|-------|--------|--------|-------------|------------|------------|
| Cascade RCNN             | Res2Net-101 + DcnV2  | sabl | PAFPN + Dyhead | 0.652 | 0.800 | 0.808  | 0.762  | 0.677       | 0.692      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
| YOLOX_m                  | YOLOX_M + CSPDarknet | -    | PAFPN          | 0.621 | 0.755 | 0.902  | 0.835  | 0.711       | 0.796      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
| Cascade RCNN             | Res2Net-50 + DcnV2   | sabl | Dyhead         | 0.680 | 0.802 | 0.820  | 0.779  | 0.704       | 0.628      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
| Ensemble Model |    |  |  | 0.753 | 0.699 | 0.863   | 0.796  | 0.717       | 0.675      ||


## Bacteria Feature Extraction

We apply the following approach to extract bacteria features:

- **Feature Pyramid Network (FPN)** for automatic color manipulation
- **Deep MAC** for instance segmentation from the bounding box given by object detection models
- **Open-CV** for features extraction 

We compare downstream cell classification tasks using features extracted from our model and CellProfiler (traditional method).

|               | Bacteria detection model | CellProfiler |
|---------------|--------------------------|--------------|
| Mean F1-score | 0.76                     | 0.796        |

This difference is considered acceptable because the number of bacteria that the CellProfiler is able to detect is significantly lower than the model’s but higher in terms of quality since the CellProfiler can only detect complete bacteria cells.

## Dataset

The current dataset contains **900 images**: Ampicillin (100), Ciprofloxacin (100), Rifampicin (100), Tetracycline (100), Mecillinam(100), Kanamycin(100), Colistin(100), and Untreated (200). We annotated the RGB version in PNG format of the bacteria images which is easier to visualize using [labelme](https://github.com/wkentaro/labelme). Images are in TIFF format that we use for actual model training and testing.

<table style="padding:10px">
  <tr>
    <td style="text-align:center"> High intensity image </td>
    <td style="text-align:center"> Low intensity image </td>
  <tr>
    <td> 
         <img src="./Bacteria images/readme_image/high_intensity_untreat.png"  alt="1" width = 256px height = 256px >
    </td>
      
  <td>
         <img src="./Bacteria images/readme_image/low_intensity_untreat.png"  alt="1" width = 256px height = 256px >
  </td>
    
   <!--<td><img src="./Scshot/trip_end.png" align="right" alt="4" width =  279px height = 496px></td>-->
  </tr>
</table>


## Installation 

### Download and activate the environment
- Download the virtual environment from [here](https://drive.google.com/file/d/1e3J-Eg9dTtupIfhvqRHu8fAz3zuM-kXr/view?usp=sharing)
- extract the virtual environment

``` sh
cd bacteria-detection
conda activate path/to/virtual_environment
```

### Download Pre-trained Weight

We release the pretrained model weight for reproducibility purposes. You can download the weights of all models [here](https://drive.google.com/drive/folders/1S8LEIkAcTxg5MJtzbsWkIeIt-Ayp5Mzz?usp=sharing) then

- move folder checkpoints to `web-app/backend/mmdetection`
- move `color_model_checkpoints` and `deepmac_1024x1024_coco17` to `web-app/backend`

### Back-end: FastAPI

```
$ cd webapp/backend
$ uvicorn app:app --reload
```

### Front-end: ReactJS

Install NodeJS [here](https://nodejs.org/en)
```
$ cd webapp/frontend/bacteria-app
$ npm start
```

### Web application

<table style="padding:10px">
  <tr>
    Upload a bacteria file(s)
    <img src="./Bacteria images/readme_image/webapp_only.png"/>
  <tr>
    Perform prediction: bbox with class (left) and bbox with index number for feature extraction table (right)
    <img src="./Bacteria images/readme_image/webapp_img.png"
  <tr>
    Table of Feature extraction results
    <img src="./Bacteria images/readme_image/webapp_table.png"/>
</table>

## Members

- Korrawiz Chotayapa
- Thanyatorn Leethamchayo
- Piraya Chinnawong
- Titipat Achakulvisut

## ACKNOWLEDGMENT
We would like to thank Poochit Nonejuie Ph.D. and Mr. Thanadon Samernate from the Institute of Molecular Biosciences that inspired us and prepared the dataset for this study.
