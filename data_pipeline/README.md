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

## Mouth / yawning data (NITYMED)
Dataset: NITYMED (Kaggle, `nikospetrellis/nitymed`, HDTV720), 126 videos available in the Kaggle copy
(107 Yawning, 19 Microsleep). Licence is CC BY-ND 4.0, so extracted frames, crops, features and split
CSVs derived from it are **not committed**; regenerate them with the scripts below.

- `extract_frames.py`: per-video CSV of EAR, MAR and head-pose landmark coordinates (every 5th frame, 25 fps -> 5 rows/s).
- `labeling.py`: yawn = MAR > 0.25 sustained for >= 1.5 s. Negative frames are >= 2 s away from any yawn and MAR below the threshold.
- Only videos with exactly 3 detected yawn segments were used (87 of 107). The other 20 were excluded and not yet inspected.
- Result: 3,895 yawn / 3,792 no_yawn mouth crops (96x96 RGB), split by video 60/13/14 (train/val/test).
- `mouth_dataset.py`: tf.data loader (normalization to [0, 1], augmentation on train only).

### Known limitations
- Labels are rule-based (MAR threshold), not manually annotated. Visual check was on a small sample only.
- Negatives we looked at were closed mouths; talking/singing frames were not checked, so the model may confuse an open mouth with a yawn.
- Split is by video, not by driver (video names have no driver id and the dataset has about 21 drivers), so test accuracy may be optimistic.
- Microsleep videos were not labeled yet; eye-closure labels for the LSTM are still undecided.

### Reproduce the mouth dataset

    python data_pipeline/build_mouth_dataset.py --root <.../DSM_Dataset-HDTV720> --out <output dir>

Outputs (kept outside the repo, never committed): features/, mouth/{yawn,no_yawn}/,
labels/mouth_{train,val,test}.csv, labels/videos_summary.csv. The run is resumable (cache/).
It assumes 25 fps videos (as in NITYMED). A full run gave 87 of 107 videos used (3 yawn segments each) and
60/13/14 videos (5,400 / 1,010 / 1,277 crops) for train/val/test, identical to our first notebook run.
The 20 excluded videos (0, 1, 2 or 4 detected segments) have not been inspected; some may contain real short yawns.
