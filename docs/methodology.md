# Methodology & Engineering Decisions

This document explains how each stage of the pipeline was designed based on the dataset audit and intermediate experiments in [`notebooks/01_original_investigation.ipynb`](../notebooks/01_original_investigation.ipynb).

## 1. Patch Extraction Strategy

### Why Tiling Instead of Resizing?

Median filament area is `1,228 px` inside a `2048 × 2048` frame, and filaments are thin structures with median bounding-box occupancy of `24.3%`. Downsampling full-disk images to `512 × 512` or `768 × 768` would shrink small filaments to a handful of pixels. We instead preserve native pixel resolution via overlapping sliding-window patches.

### Patch Size Comparison (Cell 18)

Testing three patch sizes with 25% overlap (`overlap = patch_size // 4`) and edge-anchored final coordinates:

| Patch Size | Overlap | Stride | Patches / Image | Fully Contained Instances | Containment Rate |
|---:|---:|---:|---:|---:|---:|
| 512 × 512 | 128 | 384 | 25 | 7,448 / 8,199 | 90.84% |
| **768 × 768** | **192** | **576** | **16** | **7,944 / 8,199** | **96.89%** |
| 1024 × 1024 | 256 | 768 | 9 | 8,095 / 8,199 | 98.73% |

We selected **`768 × 768` with `192` overlap** (`positions = [0, 576, 1152, 1280]`, 16 patches per image): `96.89%` of all instances fit completely inside at least one patch, while keeping memory usage low (~4.2 GB peak VRAM at batch size 4 in FP32).

### Rejected Experiment: Strict Boundary-Crossing Patch Filtering (Cells 31–33)

An earlier dataset prototype (Cell 31) tried dropping partially-contained instances or excluding any patch that intersected an instance boundary. Auditing that strict rule across all 707 images (Cells 32–33) showed it was unworkable:

| Patch Config | Valid Patches | Instance Coverage | Images with 0 Valid Patches |
|---|---:|---:|---:|
| 768 × 768 (overlap 192) | 6,269 / 11,312 (55.42%) | 4,070 / 8,199 (**49.64%**) | 4 |
| 1024 × 1024 (overlap 256) | 3,405 / 6,363 (53.51%) | 5,163 / 8,199 (**62.97%**) | 46 |

Dropping boundary-crossing patches threw away half the dataset's filaments.

### Final Dataset Design (`SolarFilamentDataset`, Cell 34)

- Keep all 16 patches per image (`600 × 16 = 9,600` train patches, `107 × 16 = 1,712` val patches).
- Clip every instance mask to the patch window; retain any mask with `> 0` foreground pixels and recompute its tight `[x1, y1, x2, y2]` box.
- **Epoch-level single-annotator sampling**: `set_epoch(epoch)` selects one annotation record per physical image for the entire epoch during training (and deterministically `record_ids[0]` during validation), preventing contradictory multi-annotator supervision on the same image within an epoch.
- **LRU cache** (`cache_size=4`) avoids repeatedly decoding the same `2048 × 2048` JPEG and COCO polygons across patches.
- Sampled patch balance (200 random training patches, Cell 37): `71.0%` positive (mean `2.28`, median `2.0`, max `7` instances), `29.0%` negative.

## 2. Model Architecture (`mask_rcnn.py`)

- **Base model**: `torchvision.models.detection.maskrcnn_resnet50_fpn` initialized from `MaskRCNN_ResNet50_FPN_Weights.DEFAULT` (COCO).
- **Grayscale stem adaptation**: Replaced `backbone.body.conv1` (3-channel RGB) with a 1-channel `Conv2d` initialized with the channel-wise mean of the pretrained RGB weights (`old_conv.weight.mean(dim=1, keepdim=True)`).
- **Heads**: `FastRCNNPredictor` and `MaskRCNNPredictor` (`dim_reduced=256`) with `num_classes = 2` (background + filament).
- **Input normalization**: `image_mean = [0.326]`, `image_std = [0.217]` (from the 20-image GONG H-alpha pixel stats), with `min_size = max_size = 768` so patches are not rescaled internally.
- **Parameter count**: `43,916,123` total (`43,703,131` trainable under torchvision's default `trainable_backbone_layers=3`).

## 3. Training & Numerical Stability (`trainer.py`)

- **FP16 failure → FP32 training**: Smoke tests at batch sizes 1, 2, and 4 under `torch.autocast("cuda", dtype=torch.float16)` passed individual forward/backward steps (1.17 / 2.01 / 3.76 GB VRAM), but full training with FP16 produced non-finite gradients (noted in Cells 45–47). The final training pipeline runs in **pure FP32** (`use_amp=False`).
- **Stability guards**:
  - Gradient clipping at `max_norm = 1.0` (`error_if_nonfinite=True`).
  - Explicit finite-loss and finite-gradient checks per batch; aborts training if `> 1%` of batches are skipped.
- **Optimization**: `AdamW(lr=1e-4, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=12, eta_min=1e-6)`, batch size `4`, early stopping (`patience=4`, `min_delta=0.001`).
- **Validation loss**: Torchvision's Mask R-CNN returns losses only in `train()` mode. `validate_one_epoch` sets `model.train()` inside `torch.no_grad()` while explicitly freezing all `_BatchNorm` modules to `eval()` and restoring them in a `finally` block.

## 4. Tiled Inference, Merging & Submission (`inference/`, `submission/`)

For each `2048 × 2048` test image:
1. Extract 16 overlapping `768 × 768` patches and run FP32 inference in batches of 4 (`torch.inference_mode()`).
2. Filter detections by `score >= 0.50`, binarize masks at `>= 0.50`, drop masks with area `< 20 px`, and project cropped masks into global `2048 × 2048` coordinates.
3. **Duplicate merging (`merge_patch_candidates`)**: Group overlapping patch detections via Union-Find when `bbox_iou >= 0.30` OR `bbox_containment >= 0.50` (`intersection / min(area_a, area_b)`). Each connected component's masks are combined via logical OR and assigned the component's maximum confidence score.
4. **Overlap resolution (`make_masks_exclusive`)**: Sort merged instances descending by score; iteratively subtract already-claimed pixels (`mask & ~claimed`) and drop masks falling below `20 px`.
5. **COCO RLE encoding & validation**: Encode each exclusive mask in Fortran-contiguous `uint8` via `pycocotools.mask.encode`, write `submission.csv` (`filament_id`, `segmentation_rle`), then decode every row and verify dimensions, area `>= 20`, and `0` overlapping pixels per image.
