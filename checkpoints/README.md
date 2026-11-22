# Model Checkpoints

Trained model weights (`*.pth`) are excluded from Git via `.gitignore` because Mask R-CNN ResNet-50 FPN checkpoints are ~170 MB (weights only) to ~500 MB (with AdamW optimizer state).

## Current Availability

The checkpoint from the original Colab A100 training run (`/kaggle/working/solar_checkpoints/best_model.pth`, Epoch 1, validation loss `0.41655`) is **not** included in this repository.

## How to Obtain or Place Checkpoints

1. **Train from scratch** using the competition dataset:
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```
   This saves `checkpoints/best_model.pth`, `checkpoints/latest_model.pth`, and `checkpoints/training_history.json`.

2. **Use an existing trained checkpoint**:
   Place your `.pth` file at `checkpoints/best_model.pth` (or pass `--checkpoint /path/to/model.pth` to `scripts/predict.py`).

## Supported Checkpoint Formats

`solar_filament.models.mask_rcnn.extract_state_dict` accepts:
- Training checkpoints produced by `scripts/train.py` (dicts containing `"model_state_dict"` alongside `"epoch"`, `"best_val_loss"`, `"optimizer_state_dict"`, `"scheduler_state_dict"`, `"history"`, `"patience_counter"`).
- Dicts keyed by `"model_state"`, `"state_dict"`, or `"model"`.
- Raw `state_dict` files (including `DataParallel` checkpoints with `module.` key prefixes, which are stripped automatically).

Do **not** substitute COCO-pretrained weights for inference on solar filaments: the classification and mask heads are randomly initialized for 2 classes (`background`, `filament`) and require training on the MAGFiLO dataset.
