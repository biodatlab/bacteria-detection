# Automatic detection and morphological feature extraction of E.Coli under antibiotic treatments

Antibiotics are the primary drug for treating various kinds of infections occurring from bacteria and microbes.
They work mainly by blocking the vital pathway of those organisms and stopping them from multiplying. Previous
research shows that we can predict the antibiotics used on bacteria by visualizing their morphology. Here,
we present object detection for detecting bacteria and identifying the antibiotics used on them with their mophological features for example DNA intensity, contour area, and min areaRect.

There are 8 classes we are interested including bacteria treated with **Ampicillin**, **Ciprofloxacin**, **Rifampicin**,
**Tetracycline**, **Mecillinam**, **Kanamycin**, **Colistin**, and **Untreated**.

The processes are divided into 2 parts
1. **Object detection**
we purpose 
* **Cascade R-CNN object detection as a base model and modified with Res2Net50 + DCNv2 backbone, PaFPN with Dyhead neck, and SABL detection head**

* **Cascade R-CNN object detection as a base model and modify with Res2Net101 + DCNv2 backbone, PaFPN with Dyhead neck and SABL detection head**

* **YoloX with PaFPN neck**

and ensemble 3 models using **YoloX with PaFPN neck**

2. **Feature extraction model**
* **Feature Pyramid Network (FPN)** for automatic color manipulation
* **Deep Mask-heads Above CenterNet (Deep-MAC)** for instance segmentation from the bounding box given by object detection models
* **Open-CV** for features extraction 

<img src="./Bacteria images/readme_image/diagram.jpg"/>

## Dataset

The current dataset contains **900 images**: Ampicillin (100), Ciprofloxacin (100), Rifampicin (100), Tetracycline (100), Mecillinam(100), Kanamycin(100), Colistin(100), 
and Untreated (200).

## Annotation tool

We annotated RGB version in PNG format of the bacteria images which is easier to visualize using [labelme](https://github.com/wkentaro/labelme).
Actual images are in TIFF format that we use for actual model training and testing.


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

## Results

The current results are as follows.\
Individual model
| Model                    | Backbone             | Head | Neck           | mAP   | mIOU  | AP(50) | AP(75) | AP (medium) | AP (large) | Checkpoint|
|--------------------------|----------------------|------|----------------|-------|-------|--------|--------|-------------|------------|------------|
| Cascade RCNN             | Res2Net-101 + DcnV2  | sabl | PAFPN + Dyhead | 0.652 | 0.800 | 0.808  | 0.762  | 0.677       | 0.692      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
| YOLOX_m                  | YOLOX_M + CSPDarknet | -    | PAFPN          | 0.621 | 0.755 | 0.902  | 0.835  | 0.711       | 0.796      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
| Cascade RCNN             | Res2Net-50 + DcnV2   | sabl | Dyhead         | 0.680 | 0.802 | 0.820  | 0.779  | 0.704       | 0.628      | [ckpt](https://drive.google.com/file/d/1gw203zflhT_YrlB67rCT4O7hIh1N6njo/view?usp=sharing)|
|Weight box fusion |    |  |  | 0.753 | 0.699 | 0.863   | 0.796  | 0.717       | 0.675      ||

Comparing features extracted from our model with CellProfiler (traditional method) by perform classification on both features
|               | Bacteria detection model | CellProfiler |
|---------------|--------------------------|--------------|
| Mean F1-score | 0.76                     | 0.796        |


## Installation 

### Download and activate the environment
- Download the virtual environment from [here]()
- extract the virtual environment

```
$ cd bacteria-detection
```
```
$ conda activate path/to/virtual_environment
```

### Download Weight
Download weight of all models [here](https://drive.google.com/drive/folders/1S8LEIkAcTxg5MJtzbsWkIeIt-Ayp5Mzz?usp=sharing)
 - move folder checkpoints to web-app/backend/mmdetection
 - move color_model_checkpoints and deepmac_1024x1024_coco17 to web-app/backend

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
Image of the web application with predicted results
<table style="padding:10px">
  <tr>
<img src="./Bacteria images/readme_image/webapp_only.png"/>
  <tr>
<img src="./Bacteria images/readme_image/webapp_img.png"
  <tr>
<img src="./Bacteria images/readme_image/webapp_table.png"/>
</table>

## Members

- Korrawiz Chotayapa
- Thanyatorn Leethamchayo
- Piraya Chinnawong
- Titipat Achakulvisut

## ACKNOWLEDGMENT
We would like to thank Poochit Nonejuie Ph.D. and Mr. Thanadon Samernate from the Institute of Molecular Biosciences that inspired us and prepared the dataset for this study.
