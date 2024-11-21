# Brain Tumor Segmentation Using DUCKNet on BraTS2020 Dataset

## This project implements brain tumor segmentation using the DUCKNet architecture on the BraTS2020 dataset. The aim is to accurately segment brain tumors from MRI images, which is crucial for diagnosis, treatment planning, and outcome prediction.

### Table of Contents

1. Introduction
2. Dataset
3. Data Preprocessing
4. DUCKNet Architecture
5. Training
6. Results and Comparison
7. Conclusion
8. Usage
9. References
10. Acknowledgments
11. Author
12. License


# 1.Introduction

## Brain tumors pose significant challenges in medical diagnosis and treatment. Automated segmentation of brain tumors from MRI scans can greatly assist medical professionals by providing consistent and objective assessments. This project leverages the DUCKNet architecture, a variant of U-Net with dense connectivity, to perform semantic segmentation on MRI images from the BraTS2020 dataset.

### Dataset

### BraTS2020 Training Data

**Description:** The Brain Tumor Segmentation (BraTS) dataset provides MRI scans along with expert annotations for tumor regions.
**Data Size**: Approximately 7 GB.
**Download Instructions:**
Ensure you have a Kaggle account.
Place your **kaggle.json API token** in the working directory.

Run the following commands to download and unzip the dataset:
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d awsaf49/brats2020-training-data
!unzip -qq brats2020-training-data.zip -d brats2020 ```
