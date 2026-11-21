# Eye State CNN (Member 2)

Binary classifier for eye state from a 64×64 grayscale eye crop: **0 = closed, 1 = open**.
**Final model: Baseline CNN (threshold 0.5)**: 97.46% test accuracy, 96.1% recall on closed eyes.

## Files
| Notebook | What it does |
|---|---|
| `eye_cnn_baseline.ipynb` | CNN built from scratch (final model) |
| `eye_mobilenetv2_finetune.ipynb` | Transfer learning: MobileNetV2 (ImageNet) + fine-tuning |
| `eye_cnn_tuning.ipynb` | Hyperparameter tuning with Keras Tuner (Hyperband, 30 trials) + threshold tuning |

Each notebook is saved with its outputs (metrics, training curves, confusion matrices).
All three ran on Kaggle (GPU T4 x2) using the MRL Eye Dataset input.

## Data
MRL Eye splits from `data_pipeline/splits` (grouped by subject: 25 / 6 / 6 subjects,
55,330 / 23,502 / 6,066 images), loaded with `data_pipeline/dataset.py`
(grayscale, 64×64, normalization 0–1, augmentation on train only).

## Results (test set: 6,066 images, 6 unseen subjects)
| Model | Val loss | Accuracy | AUC | Recall closed | Missed closed | False alarms | Params |
|---|---|---|---|---|---|---|---|
| **Baseline CNN** (t=0.5) | **0.0655** | **97.46%** | **0.997** | 96.1% | 126 | 28 | 304K |
| Tuned CNN (t=0.5) | 0.0680 | 96.24% | 0.995 | 93.7% | 204 | 24 | 304K |
| Tuned CNN (t=0.75) | 0.0680 | 96.65% | 0.995 | **96.5%** | **112** | 91 | 304K |
| MobileNetV2 + fine-tuning | 0.0903 | 95.55% | 0.991 | 93.7% | 203 | 67 | 2.26M |

Baseline confusion matrix (rows = true, cols = predicted):
|  | closed | open |
|---|---|---|
| **closed** | 3096 | 126 |
| **open** | 28 | 2816 |

### Why the baseline was chosen
- Lowest validation loss, so the choice is justified on val and not only on test.
- MobileNetV2 was weaker because of the domain gap: ImageNet is color natural images, MRL is grayscale infrared eye crops.
- Hyperband picked almost the same architecture as the baseline (32 filters, 128 dense units), so tuning gave no gain.
- Raising the decision threshold increases recall on closed eyes at the cost of more false alarms; it can be tuned later in the real-time pipeline without retraining.

## Baseline model
3 conv blocks (32 → 64 → 128 filters; Conv-BN-ReLU ×2, MaxPool, Dropout) → GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1, sigmoid).
L2 = 1e-4, Adam (lr 1e-3), binary cross-entropy, EarlyStopping (patience 5), ReduceLROnPlateau, ModelCheckpoint. Stopped at epoch 20.

## Use in real time
```python
from tensorflow import keras
from face_utils import preprocess_eye
model = keras.models.load_model("eye_cnn_baseline.keras")
p_open = model.predict(preprocess_eye(eye_roi_bgr)[None], verbose=0)[0, 0]
eye_closed = p_open < 0.5
```
Model weights (`.keras`, `.weights.h5`) are in the Kaggle output of the `eye-CNN-training` notebook and will be added here.

## Note: first run failed (fixed)
`mrl_train.csv` is sorted by label and the loader's shuffle buffer (10,000) is smaller than the train set (55,330),
so most batches contained a single class. BatchNorm learned per-class batch statistics and the model predicted
"open" for every val/test image (test accuracy 46.9%). The notebooks shuffle the full train CSV before building
the dataset. Suggested permanent fix in `data_pipeline/dataset.py`: `ds.shuffle(len(df), ...)`.
