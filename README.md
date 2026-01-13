# DeepFake Detection
# DeepFake Detection using Deep Learning

This project focuses on detecting DeepFake videos by classifying facial frames as
real or fake using deep learning models. The work is based on the FaceForensics++
dataset (C23 compression) and follows a consistent preprocessing and evaluation
pipeline.

## Project Overview
- Frames are extracted from videos at fixed intervals to reduce redundancy
- Faces are detected using MTCNN and resized for model-specific input requirements
- Extracted frames are used to train and evaluate CNN-based classifiers
- Model performance is compared using standard classification metrics

## Models Used
- ResNet50 (primary model implemented and trained by me)
- EfficientNet-B3 (used for comparative analysis)
- XceptionNet
- DenseNet121

> Note: Additional architectures discussed in the report are part of the broader
> study but are not included as executable code in this repository.

## My Contribution
- Designed and implemented the frame and face extraction pipeline
- Organized real and fake frame datasets for training
- Implemented and fine-tuned the ResNet50 and DenseNet121-based classification model
- Evaluated results and compared performance with other architectures

## Dataset
- FaceForensics++ (C23 compression)
- Binary classification: real vs fake
- Due to size constraints, videos and extracted frames are not uploaded

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV, MTCNN
- NumPy, scikit-learn

## Reference
This repository is based on the experimental study and results documented in the
project report included in the `report/` directory.
