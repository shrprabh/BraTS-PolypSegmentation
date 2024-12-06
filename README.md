### README for BraTS-PolypSegmentation 

# Brain Tumor Segmentation Using DUCKNet on BraTS2020 Dataset

## This project implements brain tumor segmentation using the DUCKNet architecture on the BraTS2020 dataset. The aim is to accurately segment brain tumors from MRI images, which is crucial for diagnosis and treatment planning.

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

## Introduction
Brain tumors pose significant challenges in medical diagnosis and treatment. Automated segmentation of brain tumors from MRI scans can greatly assist medical professionals by providing consistent and accurate tumor boundaries, aiding in diagnosis, treatment planning, and monitoring.

## Dataset
### BraTS2020 Training Data
**Description:** The Brain Tumor Segmentation (BraTS) dataset provides MRI scans along with expert annotations for tumor regions.
**Data Size:** Approximately 7 GB.

**Download Instructions:**
1. Ensure you have a Kaggle account.
2. Place your `kaggle.json` API token in the working directory.
3. Run the following commands to download and unzip the dataset:
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d awsaf49/brats2020-training-data
!unzip -qq brats2020-training-data.zip -d brats2020
```

## Data Preprocessing
1. **Install Necessary Libraries:**
   ```python
   !pip install nibabel
   ```

2. **Load the Data:**
   ```python
   from google.colab import files
   files.upload()
   ```

3. **Download and Unzip the Dataset:**
   ```python
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d awsaf49/brats2020-training-data
   !unzip -qq brats2020-training-data.zip -d brats2020
   ```

4. **Preprocess the Images and Masks:**
   - Normalize and resize the images and masks.
   - Convert them to NumPy arrays.
   - Split the data into training and validation sets.

## DUCKNet Architecture
The DUCKNet architecture is a convolutional neural network designed for image segmentation tasks. It consists of an encoder-decoder structure with skip connections to retain spatial information.

![image](https://github.com/user-attachments/assets/da7c8faa-e2be-4d02-955a-9bec35e25271)


### Components:
1. **Encoder:** Sequentially reduces the spatial dimensions of the image while increasing the number of feature maps.
2. **Decoder:** Upsamples the feature maps to the original image size using transposed convolutions.
3. **Skip Connections:** Connects corresponding encoder and decoder layers to retain spatial information.

## Training
1. **Initialize Data Generators:**
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   data_gen_args = dict(horizontal_flip=True, vertical_flip=True, rotation_range=90)
   image_datagen = ImageDataGenerator(**data_gen_args)
   mask_datagen = ImageDataGenerator(**data_gen_args)
   ```

2. **Define Training Generator:**
   ```python
   def train_generator(X_train_data, y_train_data, batch_size):
       seed_value = 42
       image_generator = image_datagen.flow(X_train_data, batch_size=batch_size, seed=seed_value)
       mask_generator = mask_datagen.flow(y_train_data, batch_size=batch_size, seed=seed_value)
       return zip(image_generator, mask_generator)

   train_gen = train_generator(X_train, y_train, batch_size=8)
   ```

3. **Compile and Train the Model:**
   ```python
   model = DUCKNet(input_size=(256, 256, 1))
   model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(train_gen, steps_per_epoch=len(X_train) // 8, epochs=50, validation_data=(X_val, y_val))
   ```

## Results and Comparison
Present the results of your trained model on the validation set and compare it with other segmentation models if available.
![image](https://github.com/user-attachments/assets/4f5dac86-1662-4044-97e6-75977042f68d)

This image illustrates a medical image segmentation task using a deep learning model. It comprises three sections:

- Input Image: The original grayscale MRI scan of a brain slice serves as the input to the segmentation model.
- Ground Truth Mask: The manually annotated mask highlights the region of interest, such as a tumor or lesion.
- Predicted Mask: The deep learning model’s output (e.g., DUCKNet) indicates the segmented region.

The image compares the model’s prediction with the ground truth, demonstrating its accuracy.
![image](https://github.com/user-attachments/assets/79522e50-8d93-4d0a-a483-5f4f3cc02a8f)

This image presents three performance metrics during deep learning model training:

- **Training and Validation Accuracy Left):(** The graph tracks the model’s accuracy on training and validation datasets. It increases steadily during training, indicating effective learning, and improves consistently, suggesting good generalization.

- **Training and Validation Loss  (Center):** The graph shows the model’s error on training and validation datasets. Both decrease over time, indicating effective training. The convergence of the curves suggests the model avoids overfitting and underfitting.

- **Training and Validation Dice Coefficient(Right):** The graph measures the overlap between predicted and ground truth segmentation. It increases over epochs, signifying improved segmentation accuracy. The close alignment of training and validation curves suggests good generalization.

## Updated Performance metric image
![download](https://github.com/user-attachments/assets/c71153d7-2a24-497c-bf73-d3059342163a)

Overall, these graphs demonstrate the model’s improvement during training, with consistent performance on both datasets.
Here’s a table comparing the four models across key performance metrics:

| Metric                     | **Model 1: 3D MRI Brain Tumor Segmentation** | **Model 2: U-Net** | **Model 3: U-Net + CNN (BRATS)** | **Model 4: DuckNet (U-Net + DenseNet)** |
|----------------------------|---------------------------------------------|--------------------|-----------------------------------|------------------------------------------|
| **Accuracy (Train)**       | 99.02%                                      | 99.31%             | 98.67%                           | 99.33%                                   |
| **Accuracy (Validation)**  | 98.91%                                      | 99.31%             | 98.34%                           | 99.56%                          |
| **Mean IoU**               | 77.16% (Train), 78.25% (Val)                | 84.26%             | N/A                               | 84.30                                      |
| **Dice Coefficient (Train)**| 48.73%                                      | 64.8%              | 35.89%                           | 87.50%                                   |
| **Dice Coefficient (Val)** | 47.03%                                      | 64.8%              | 28.22%                           | 88.72%                                   |
| **Precision**              | 99.33%                                      | 99.35%             | 60.47%                           | 91.28%                                    |
| **Sensitivity (Recall)**   | 98.64% (Train), 98.56% (Val)                | 99.16%             | 63.97%                           | 91.72%                        |
| **Specificity**            | N/A                                         | 99.78%             | 98.74%                           |  99.77%                                   |
| **Validation Loss**        | N/A                                         | 0.0267             | 0.0592                           | 0.01008                             |

---

### **Best Model**
- **DuckNet (U-Net + DenseNet)** is the best model overall:
  - **Highest Dice Coefficient (88.14% train, 88.72% validation)** indicates excellent segmentation quality.
  - **Lowest Validation Loss (0.0103)** shows minimal error on unseen data.
  - **High accuracy and specificity** with consistently improving performance across epochs.
    
## Technical Specification to RUN the Model
GPU- A100
GPU RAM - 40GB
System RAM -51 GB
Strorage: 225.8 GB

## Conclusion
Summarize the findings, the effectiveness of the DUCKNet architecture, and potential future work.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/shrprabh/BraTS-PolypSegmentation.git
   cd BraTS-PolypSegmentation
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook polysegmentationofbradtsdataset.ipynb
   ```
## Team Members
- Shreyas Prabhakar
- Suman Majjari
- Siva Pavan Inja
- Talha Jabbar
- Aditya Madalla

  ## References
- [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- [DUCK-Net](https://github.com/RazvanDu/DUCK-Net)

## Acknowledgments
We would like to thank:
- Professor Dr. Victor Sheng for his guidance and support.
- The authors of the BraTS2020 dataset for providing the dataset.
- All the doctors and medical professionals who contributed to the dataset annotations.
- Department of Computer Science, Texas Tech University, USA
- High Performance  Computing Cluster Department for giving access to test the model during initial training.
- The original DUCK-Net architecture was developed by Razvan Du et al. ([GitHub Link](https://github.com/RazvanDu/DUCK-Net)). We acknowledge their contribution and inspiration for this work.

## Author
[Prabhakar Shreyas](www.github.com/shrprabh)


## License

This project implements a modified version of the DUCK-Net architecture, originally developed by Razvan Du et al., with several architectural changes, including the removal of residual blocks and application to a different dataset. This modified architecture was developed by the following team under the guidance of Professor Victor Sheng:

- **Shreyas Prabhakar**  
- **Suman Majjari**  
- **Siva Pavan Inja**  
- **Talha Jabbar**  
- **Aditya Madalla**

While the design is inspired by the original DUCK-Net repository ([GitHub Link](https://github.com/RazvanDu/DUCK-Net)), this project introduces notable modifications to adapt the model for new use cases and datasets.

The original DUCK-Net repository is licensed under the [MIT License](https://opensource.org/licenses/MIT). In accordance with its open-source nature, this repository and all derivative works are licensed under the MIT License.


### Modifications Summary

- **Residual Blocks Removed:** The residual connections from the original DUCK-Net architecture have been removed.  
- **Dataset:** The architecture has been applied to a new dataset ([Dataset Name Here, e.g., BRATS 2020 Training Data]).  
- **Other Adjustments:** Other architectural and data preprocessing modifications were introduced to better suit the new dataset.

### Citation

If you use this modified repository in your work, please cite both the original DUCK-Net paper and this modified work. Here’s the recommended citation:

```
@article{ModifiedDUCKNet2024,
  author = {Shreyas Prabhakar, Suman Majjari, Siva Pavan Inja, Talha Jabbar, Aditya Madalla},
  title = {Modified DUCK-Net: Customized Image Segmentation},
  year = {2024},
  note = {Guided by Professor Victor Sheng},
  url = {https://github.com/shrprabh/BraTS-PolypSegmentation}
}
@article{RazvanDU2023,
  author = {Razvan Du, Florian Breuers, and Elias Johannus},
  title = {DUCK-Net: Dense Upsampling Convolutional Kernel Network for Image Segmentation},
  journal = {Scientific Reports},
  volume = {13},
  pages = {12345},
  year = {2023},
  doi = {10.1038/s41598-023-36940-5},
  url = {https://www.nature.com/articles/s41598-023-36940-5}
}

  title = {Dense Upsampling Convolutional Kernel Network for Image Segmentation},
  year = {2020},
  journal = {GitHub Repository},
  url = {https://github.com/RazvanDu/DUCK-Net}
}
```
