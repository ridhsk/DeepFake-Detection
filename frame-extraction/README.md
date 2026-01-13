# Frame and Face Extraction Module

This module extracts face frames from real and fake videos for deepfake detection.
It was designed to generate a clean and balanced image dataset suitable for
training deep learning models such as ResNet.

## Methodology
- Videos are processed frame-by-frame using OpenCV
- Frames are sampled at fixed intervals to reduce redundancy
- MTCNN is used to detect and crop facial regions
- Extracted faces are resized to 224×224
- A fixed number of frames are saved per video

## Dataset Structure
- real/  → frames extracted from original videos
- fake/  → frames extracted from manipulated videos

## Configuration
- Frame skip: 10
- Frames per video: 5
- Image size: 224×224

## Notes
Due to storage constraints, extracted frames are not uploaded to this repository.
They can be regenerated using the provided script.
