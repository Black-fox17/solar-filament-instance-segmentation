# Experiments, Measured Results & Limitations

This record reports only what was actually executed and measured in [`notebooks/01_original_investigation.ipynb`](../notebooks/01_original_investigation.ipynb), along with honest limitations.

## 1. Execution Environment

- **Platform**: Google Colab high-memory runtime (`A100`)
- **GPU**: NVIDIA A100-SXM4-80GB (79.25 GB VRAM, CUDA capability 8.0)
- **Software**: Python `3.13.16`, PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`

## 2. Training Run Results (Cell 46)

Configuration: FP32, `AdamW(lr=1e-4, wd=1e-4)`, `CosineAnnealingLR(T_max=12, eta_min=1e-6)`, batch size `4` (`2,400` train batches / `428` val batches per epoch), `grad_clip=1.0`, `patience=4`, `min_delta=0.001`.

| Epoch | LR | Train Loss | Val Loss | Train Mask | Val Mask | Train Obj | Val Obj | Grad Norm (Mean / Max) | Peak VRAM | Skipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **1** | `1.000e-4` | `0.45668` | **`0.41655`** | `0.23219` | `0.21231` | `0.03172` | `0.03409` | `2.201 / 101.527` | `4.14 GB` | `0 / 0` |
| 2 | `9.831e-5` | `0.38394` | `0.43285` | `0.20751` | `0.21268` | `0.02310` | `0.03572` | `1.828 / 14.243` | `4.25 GB` | `0 / 0` |
| 3 | `9.337e-5` | `0.34356` | `0.43748` | `0.19647` | `0.20681` | `0.01818` | `0.05534` | `1.836 / 13.002` | `4.18 GB` | `0 / 0` |
| 4 | `8.550e-5` | `0.31131` | `0.43854` | `0.18521` | `0.21548` | `0.01340` | `0.04631` | `1.910 / 13.354` | `4.23 GB` | `0 / 0` |
| 5 | `7.525e-5` | `0.28526` | `0.46518` | `0.17484` | `0.21891` | `0.01005` | `0.06855` | `1.868 / 10.707` | `4.40 GB` | `0 / 0` |

- **Early stopping** triggered after Epoch 5 (4 consecutive epochs without validation loss improvement).
- **Best restored checkpoint**: Epoch 1 (`val_loss = 0.41655`), total wall time `0.57 hours` (~6.5 min/epoch, ~26.6 patches/sec).
- **Numerical stability**: `0` skipped batches across all 5 epochs in FP32.

## 3. Test-Set Inference & Submission Results (Cell 47)

Using the Epoch 1 `best_model.pth` checkpoint on the 180 test images (`2048 × 2048`):
- **Raw patch detections** (`score >= 0.50`, `area >= 20`): `6,599`
- **Merged instances** (Union-Find `bbox_iou >= 0.30` or `containment >= 0.50`): `1,689`
- **Final exclusive instances**: `1,689` (`submission.csv` shape `(1689, 2)`)
- **Unique predicted test images**: `179 / 180` (`1` test image had zero detections above `0.50`)
- **Overlapping instances after RLE decode audit**: `0`

*Note on Cell 48*: Cell 48 in the original notebook prints `CSV rows: 1742` from an earlier submission file on disk before Cell 47 was re-run (which produced `1,689` rows).

## 4. Unexecuted Diagnostic Cells (Cells 49–53)

Cells 49–53 in the original notebook contain code for qualitative validation-patch visualization and a confidence-threshold sweep (`[0.05, 0.10, 0.25, 0.50]`) over 50 positive validation patches, **but have no saved execution outputs**. No quantitative threshold-sweep metrics from those cells are claimed here.

## 5. Known Weaknesses & Limitations

1. **Rapid overfitting on validation loss**: Training loss dropped steadily from `0.45668` to `0.28526`, while validation loss was lowest at **Epoch 1** (`0.41655`) and degraded every epoch thereafter (most notably validation RPN `loss_objectness`, which doubled from `0.03409` to `0.06855`). Contributing factors likely include zero data augmentation, only 600 unique physical training images, and `lr=1e-4` with AdamW on an unfrozen ResNet-50 backbone.
2. **No instance-segmentation metric evaluated**: Model selection used Mask R-CNN validation *loss* only. We did **not** compute COCO mask AP, F1, Dice, or competition leaderboard metrics on the validation split. Validation loss can diverge from mask AP (especially when RPN objectness loss rises while mask loss stays flat at ~0.207–0.219).
3. **Single-annotator validation target**: Validation always compares against `record_ids[0]` per physical image, even though inter-annotator union IoU on multi-annotator images can be as low as `0.36–0.48`.
4. **Bounding-box-based duplicate merging can falsely merge distinct filaments**: Because filaments fill a median of only `24.3%` of their axis-aligned bounding boxes, two separate nearby curved filaments can satisfy `bbox_iou >= 0.30` or `bbox_containment >= 0.50` without touching at the pixel level. Union-Find transitivity can also chain adjacent detections into a single merged instance.
5. **1 of 180 test images received 0 predictions**: At `score_threshold = 0.50`, one test image produced no rows in `submission.csv`; whether that image genuinely contains no filaments or fell just below threshold was not audited.
6. **Checkpoint not stored in Git**: The Epoch 1 checkpoint was saved inside the ephemeral Colab/Kaggle workspace (`/kaggle/working/solar_checkpoints/best_model.pth`) and is not present in this repository.

## 6. Candidate Future Improvements (Not Yet Tested)

- Add domain-appropriate augmentations (random flips, 90° rotations, mild contrast/gamma jitter).
- Implement full-resolution validation evaluation using COCO mask AP / Dice / competition IoU matching (rather than patch-level loss alone).
- Replace bounding-box Union-Find merging with **pixel-mask IoU / mask containment** (or soft-NMS / WBF on masks) to avoid false merges between curved neighboring filaments.
- Explore lower learning rates (e.g., `2e-5`–`5e-5`), warmup, or backbone freezing to curb Epoch 2+ overfitting.
- Investigate multi-annotator consensus masks (or evaluating against all available annotators on validation).
