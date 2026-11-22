# Mouth State CNN (Member 2)

Binary classifier for mouth state from a 96×96 RGB mouth crop: **0 = normal, 1 = yawn**.
**Final model: Tuned CNN (threshold 0.5)**: 98.43% test accuracy, 98.6% recall on yawns
(99.8% recall with the validation-selected threshold 0.1).

## Files
| Notebook | What it does |
|---|---|
| `mouth_cnn_v2.ipynb` | CNN from scratch with the fixes for the failed v1 (per-image standardization, lighting augmentation, BN momentum 0.9) |
| `mouth_mobilenetv2_finetune.ipynb` | Transfer learning: MobileNetV2 (ImageNet) + fine-tuning |
| `mouth_cnn_tuning.ipynb` | Hyperparameter tuning with Keras Tuner (Hyperband, 30 trials) + threshold tuning (final model) |
| `mouth_cnn_tuned.keras` | Final trained model (architecture + weights) |

Each notebook is saved with its outputs (metrics, training curves, confusion matrices).
All ran on Kaggle using the output of the `deep learning1` notebook (mouth crops built with `data_pipeline/build_mouth_dataset.py`) as input.

## Data
NITYMED mouth crops built with Member 1's script: 87 of 107 yawning videos, split by video (60 / 13 / 14).
| Split | Images | Normal | Yawn |
|---|---|---|---|
| Train | 5,398 | 2,700 | 2,698 |
| Val | 1,010 | 455 | 555 |
| Test | 1,276 | 636 | 640 |

Loaded with `data_pipeline/mouth_dataset.py` (RGB, 96×96, normalization 0–1, augmentation on train only).
The notebooks rewrite the CSV paths (`/kaggle/working/nitymed_out` → the input location) and shuffle the full train CSV.

## Results (test set: 1,276 images, 14 unseen videos)
| Model | Val loss | Accuracy | AUC | Yawn recall | Missed yawns | False alarms | Params |
|---|---|---|---|---|---|---|---|
| v1 baseline (failed) | 3.158 | 49.84% | 0.5 | 0% | 640 | 0 | 305K |
| v2 CNN | 0.0970 | **98.59%** | 0.9993 | 98.75% | 8 | 10 | 305K |
| MobileNetV2 + fine-tuning | 0.0850 | 98.43% | 0.9992 | 97.66% | 15 | **5** | 2.26M |
| **Tuned CNN, t=0.5 (final)** | **0.0637** | 98.43% | 0.9993 | 98.59% | 9 | 11 | 305K |
| Tuned CNN, t=0.1 | 0.0637 | 98.12% | 0.9993 | **99.84%** | **1** | 23 | 305K |

Final model confusion matrix (rows = true, cols = predicted):
|  | normal | yawn |
|---|---|---|
| **normal** | 625 | 11 |
| **yawn** | 9 | 631 |

### Why the tuned CNN was chosen
- Lowest validation loss, the same selection rule used for the eye model.
- Same small size as the baseline (305K parameters), 7× smaller than MobileNetV2.
- The three working models are within two test images of each other; the big gain came from fixing v1.

## Note: v1 failed (fixed in v2)
v1 reached 98% train accuracy but predicted "normal" for every val/test image (val accuracy fixed at 45%, test 49.8%).
Training data was already shuffled, so this was a different issue from the eye model: a lighting/appearance shift
between videos combined with BatchNorm statistics. v2 adds, inside the model: per-image standardization,
brightness ±0.3 / contrast ±0.4 augmentation, and BatchNorm momentum 0.9. The three fixes were applied together.

## Limitations
- Yawn labels are rule-based (MAR > 0.25 for ≥ 1.5 s), not manual; talking/singing was not tested.
- Split is by video, not by driver (~21 drivers), so test results may be optimistic.
- Hyperband used the same seed (42) as the eye tuning and returned the same configuration.

## Use in real time
```python
from tensorflow import keras
from face_utils import preprocess_mouth
model = keras.models.load_model("mouth_cnn_tuned.keras")
p_yawn = model.predict(preprocess_mouth(mouth_roi_bgr)[None], verbose=0)[0, 0]
yawning = p_yawn >= 0.5          # 0.1 maximizes yawn recall (validation-selected)
```
