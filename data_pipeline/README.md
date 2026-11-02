# Data Pipeline (Member 1)

## Files
- `face_utils.py`: MediaPipe FaceLandmarker (Tasks API), ROI cropping (eyes/mouth), EAR/MAR, preprocessing.
- `dataset.py`: tf.data pipeline (grayscale 64x64, normalization 0-1, augmentation on train only).
- `splits/`: `mrl_train.csv`, `mrl_val.csv`, `mrl_test.csv` (columns: path, label, subject).

## Dataset used
MRL Eye Dataset (Kaggle copy by `imadeddinedjerarda/mrl-eye-dataset`), 84,898 images.
The original `taufiqabs` copy was not available on Kaggle, so this copy (same image count) was used.
Labels: 0 = closed, 1 = open.

## Split (grouped by subject, no subject appears in two splits)
| split | subjects | images |
|---|---|---|
| train | 25 | 55,330 |
| val | 6 | 23,502 |
| test | 6 | 6,066 |
Not exactly 70/15/15 because there are only 37 subjects with different image counts.

## Important notes
- `mp.solutions` was removed from recent MediaPipe versions, so we use `FaceLandmarker`. Use `mediapipe>=0.10.30`.
- `face_landmarker.task` is downloaded automatically on first import (needs internet), not committed.
- CSV paths point to Kaggle (`/kaggle/input/datasets/imadeddinedjerarda/mrl-eye-dataset/mrleyedataset`).
- Eye preprocessing is grayscale because MRL images are grayscale. Use `preprocess_eye` at training and real-time.

## Usage
```python
from dataset import make_dataset
train_ds = make_dataset("splits/mrl_train.csv", training=True)
```
