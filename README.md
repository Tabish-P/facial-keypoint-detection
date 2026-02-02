# Facial Keypoint Detection

A deep learning project that detects facial keypoints (landmarks) in images using Convolutional Neural Networks (CNN). The model identifies 68 facial keypoints on detected faces, enabling applications in face recognition, emotion detection, and facial animation.

## Project Overview

This project implements a complete pipeline for facial keypoint detection:

1. **Face Detection** - Locates faces in images using Haar Cascade classifiers
2. **Face Preprocessing** - Converts faces to grayscale, normalizes, and resizes to model input dimensions
3. **Keypoint Detection** - Uses a trained CNN to predict 68 facial keypoints per face
4. **Visualization** - Displays detected keypoints on the original image

## Features

- CNN-based facial keypoint detection (68 landmarks)
- Face detection using OpenCV's Haar Cascade
- Data loading and preprocessing pipeline
- Model training and evaluation
- Real-time keypoint detection on images
- Visualization of detected keypoints

## Project Structure

```
facial-keypoint-detection/
├── data_load.py                              # Dataset class and data loading utilities
├── models.py                                 # CNN architecture definition
├── network-architecture.ipynb                # Network design exploration
├── load-visualize-data.ipynb                 # Data loading and visualization
├── facial-keypoint-detection-pipeline.ipynb  # Complete detection pipeline
├── saved_models/                             # Trained model weights
│   └── keypoints_model_1.pt
├── detector_architectures/                   # Haar Cascade classifiers
│   └── haarcascade_frontalface_default.xml
└── images/                                   # Test images for detection
```

## Model Architecture

The CNN architecture consists of:

- **4 Convolutional Layers** - Feature extraction with increasing depth (32 → 64 → 128 → 256 channels)
- **Max Pooling Layers** - Spatial dimensionality reduction
- **3 Fully Connected Layers** - Prediction of 136 values (68 keypoints × 2 coordinates)
- **Dropout Regularization** - Prevents overfitting

**Input**: Grayscale face image (1 × 224 × 224)
**Output**: 68 facial keypoints (136 values: x, y coordinates)

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd facial-keypoint-detection
```

2. Install dependencies:
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib
```

3. Download the pre-trained model or train your own using the provided notebooks.

## Usage

### Quick Start - Detect Keypoints on an Image

Open `facial-keypoint-detection-pipeline.ipynb` and follow these steps:

1. **Load an image** - Select an image containing faces from the `images/` directory
2. **Detect faces** - The Haar Cascade detector identifies all faces
3. **Load the model** - Load a pre-trained model from `saved_models/`
4. **Predict keypoints** - The CNN predicts all 68 facial keypoints
5. **Visualize results** - See the keypoints overlaid on the original image

### Training a Model

Use `network-architecture.ipynb` to:
- Explore different network architectures
- Train the model on facial keypoint datasets
- Evaluate performance metrics

### Data Loading

The `data_load.py` module provides:
- `FacialKeypointsDataset` - Custom PyTorch Dataset class
- Data preprocessing and augmentation
- DataLoader utilities for batch processing

Use `load-visualize-data.ipynb` to:
- Explore the dataset
- Visualize training samples
- Understand keypoint annotations

## Model Performance

The trained model achieves accurate keypoint detection across diverse facial poses and expressions. Performance can be evaluated on validation sets to ensure robustness.

## Key Points Detected

The model detects 68 facial keypoints including:
- **Eyes** - Corners and centers of both eyes
- **Eyebrows** - Upper and lower contours
- **Nose** - Bridge and tip
- **Mouth** - Corners, center, and outline
- **Face** - Jawline and cheek points

## Limitations

- Model trained on specific dataset characteristics
- Performance may vary with different lighting conditions
- Requires reasonably clear face images
- Single face detection preferred (handles multiple faces but best with clear face regions)

## Future Improvements

- 3D facial keypoint detection
- Real-time video processing
- Multi-face optimization
- Model ensemble methods
- Transfer learning from larger datasets

## Acknowledgments

This project was completed as part of the Udacity Computer Vision Nanodegree Program.

## Author

Tabish Punjani

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and suggestions.
