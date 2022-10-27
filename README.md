# Object detection of bacteria treated with antibiotics from microscopic images

Antibiotics are the primary drug for treating various kinds of infections occurring from bacteria and microbes.
They work mainly by blocking the vital pathway of those organisms and stopping them from multiplying. Previous
research shows that we can predict the antibiotics used on bacteria by visualizing its morphology. Here,
we present the object detection for detecting bacteria and identifying the antibiotics used on them.
There are 5 classes we are interested including bacteria treated with **Ampicillin**, **Ciprofloxacin**, **Rifampicin**,
**Tetracycline**, and **Untreated**.

## Dataset

Current dataset contains **600 images**: Ampicillin (100), Ciprofloxacin (100), Rifampicin (100), Tetracycline (100),
and Untreated (100).

## Annotation tool

We annotated RGB version in PNG format of the bacteria images which is easier to visualize using [labelme](https://github.com/wkentaro/labelme).
Actual images are in TIFF format that we use for actual training and testing.

## Members

- Ong
- Tangkwa
- Ploy
- Titipat Achakulvisut
