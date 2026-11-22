# Dataset Analysis (MAGFiLO 1.0 — Kaggle 2026)

Everything in this document comes directly from the exploratory data analysis in [`notebooks/01_original_investigation.ipynb`](../notebooks/01_original_investigation.ipynb) (Cells 0–30).

## 1. Dataset Layout & Image Properties

The competition dataset (`filament-segmentation-2026`, 671 MB archive via `kagglehub`) unpacks into `MAGFiLO_1.0_Kaggle_2026/`:

| Split | Directory / File | Count / Size |
|---|---|---|
| Train images | `train/train_images/*.jpeg` | 707 physical `.jpeg` images |
| Train annotations | `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` | 46.36 MB COCO JSON (1,154 image records, 8,199 annotations) |
| Test images | `test/test_images/*.jpeg` | 180 physical `.jpeg` images |

All images are full-disk GONG H-alpha solar observations (NSO URLs in metadata, e.g., `https://gong2.nso.edu/HA/hag/...`):
- **Resolution**: `2048 × 2048` across all 1,154 training records and sampled test images.
- **Color mode**: Single-channel 8-bit grayscale (`L`, JPEG).
- **Pixel statistics** (measured across 20 sampled training images):
  - Mean: `83.09` (`0.326` after `/255`)
  - Std: `55.40` (`0.217` after `/255`)
  - Average min / max: `0.00` / `239.50`
- **File resolution**: All 707 unique filenames in the training JSON resolve to existing files in `train/train_images/` (0 missing).

## 2. The Multi-Annotator Trap (1,154 Records vs. 707 Physical Images)

The most consequential finding in the EDA was that the COCO `images` list contains **1,154 records**, but only **707 unique `file_name` entries**.

| Annotation Records per Physical Image | Number of Physical Images |
|---|---:|
| 1 record | 411 |
| 2 records | 145 |
| 3 records | 151 |
| **Total** | **707 unique images (1,154 records)** |

Records sharing a physical filename have distinct COCO `id` prefixes (e.g., `050101-20111116063134Lh`, `050102-20111116063134Lh`, `050103-20111116063134Lh`) and represent separate annotator passes over the exact same solar disk image.

### Inter-Annotator Disagreement

Comparing union foreground masks across annotators on the same physical image revealed substantial disagreement:
- On `20111116063134Lh.jpeg`, all three annotators marked 11 instances, yet total foreground pixels varied from `28,884` to `29,875` to `45,522`.
- On `20131007075314Th.jpeg` (3 records), pairwise union mask IoUs were only:
  - `010403` vs `010402`: **0.4778**
  - `010403` vs `010401`: **0.4009**
  - `010402` vs `010401`: **0.3604**

### Leakage Prevention

A naive `train_test_split` over the 1,154 COCO record IDs (Cell 8: 980 train / 174 val) would place the same physical solar images into both training and validation. Switching to a **physical-filename split** (Cell 11, 85/15, `random_state=42`) produced:
- **Train**: 600 unique physical images (992 annotation records)
- **Validation**: 107 unique physical images (162 annotation records)
- **Filename overlap**: 0 (asserted)

## 3. Categories & Instance Counts

All 8,199 annotations are stored as COCO polygon lists with positive area (`0` non-positive areas, `0` records with zero annotations).

| Category ID | Name | Supercategory | Annotations | Share | Median Area (px) | Median BBox (W × H) |
|---:|---|---|---:|---:|---:|---:|
| 1 | `Left` | `filament` | 2,535 | 30.92% | 1,438.0 | 87.30 × 85.44 |
| 2 | `Right` | `filament` | 2,590 | 31.59% | 1,524.0 | 92.35 × 84.44 |
| 3 | `Unidentifiable` | `filament` | 3,074 | 37.49% | 868.5 | 55.21 × 50.48 |
| 4 | `Ambiguous` | `filament` | 0 | 0.00% | — | — |

Instances per image record: min `1`, median `7.0`, mean `7.10`, max `26`.

Because the competition evaluates class-agnostic filament instance segmentation (`filament_id`, `segmentation_rle`), categories 1–3 are collapsed into a single foreground class (`1`).

## 4. Filament Geometry & Morphology

Filaments are sparse, thin, and frequently curved or elongated:

| Metric | Median | 75th Percentile | 90th Percentile | Max |
|---|---:|---:|---:|---:|
| Mask area (px) | 1,228.0 | — | 4,684.6 | 37,739.0 |
| BBox width (px) | 74.52 | 126.21 | 202.48 | 1,067.31 |
| BBox height (px) | 69.61 | 116.59 | 186.51 | 709.81 |
| Aspect ratio | 1.66 | 2.36 | 3.23 | 12.78 |
| BBox occupancy (`mask / bbox`) | **24.30%** | 36.15% | 47.79% | 87.54% |

- `36.10%` of instances have aspect ratio ≥ 2, `12.86%` ≥ 3, and `1.66%` ≥ 5.
- A median filament covers only **0.029%** of a `2048 × 2048` image (`1,228 / 4,194,304` pixels) and fills less than a quarter of its axis-aligned bounding box.

## 5. Mask Integrity & Intra-Record Overlap Audit

Auditing all 1,154 records (8,199 masks) in Cells 22–25 showed:
- **Empty masks**: 0
- **Invalid / unreadable masks**: 0
- **Records with any pixel overlap between instances**: 5 out of 1,154

Of those 5 records:
1. `010403-20160810234034Lh` (`20160810234034Lh.jpeg`): Two `Right` annotations (`5,874 px` and `5,959 px`, intersection `4,970 px`, **IoU = 0.7242**) — the only pair in the entire dataset with IoU ≥ 0.50 (almost certainly an annotator duplicate).
2. `050202-20120523163014Mh`: 60 px mask completely inside a 1,229 px mask (IoU `0.0488`).
3. `010403-20150228082414Th`: 9 px mask completely inside a 1,745 px mask (IoU `0.0052`).
4. `050102-20150116082034Lh` and `030101-20210815140550Bh`: 1-pixel boundary touches.

Ground-truth filament instances are therefore effectively mutually exclusive, consistent with the competition's non-overlapping submission requirement.
