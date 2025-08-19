# Model Files

This folder contains the pre-trained model weights, configuration files, and scripts required for converting the trained model into the CoreML format used in the ElifPose iOS app.

## Contents

### 1. `ElifPose.pth`
- This file contains the **pre-trained model weights** for the pose estimation model used in the app.
- The detection model's weights are **not included** here, as they remain unchanged from the original model provided by the [MMPose](https://github.com/open-mmlab/mmpose) library.

### 2. `CoreMLExporter.ipynb`
- This Jupyter notebook sets up the **conversion pipeline** to convert the PyTorch model (`.pth` file) into **CoreML** mlpackage format.
- It downloads all the required libraries and automates the conversion process, making it easy to integrate the model into the iOS app.
- The notebook will also install the detection model by itself, so no additional steps are needed for the detection component.

### 3. `elifpose_8b1024`
- This file contains the **model configuration** used to train the pose estimation model.
- It defines the architecture and hyperparameters used during training.

### 4. `rtmdet_nano`
- This file contains the configuration for the **detection model**. 
- The detection model is automatically installed and configured by the `CoreMLExporter.ipynb` notebook.

## How to Use

1. Open the `CoreMLExporter.ipynb` notebook.
2. Follow the instructions inside the notebook to set up the required environment and convert the `ElifPose.pth` model into the CoreML format.
3. The detection model will be automatically downloaded and configured during the conversion process.


## Acknowledgments

The detection and pose estimation model is based on the [MMPose](https://github.com/open-mmlab/mmpose) library, and the conversion tools and techniques used in this folder rely on various open-source tools. The notebook is adapted from discussion in the github issues for mmdeploy library [Converter](https://github.com/open-mmlab/mmdeploy/issues/2794) .