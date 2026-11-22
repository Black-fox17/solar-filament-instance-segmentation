"""Dataset, annotation parsing, and patch geometry modules."""

from solar_filament.data.annotations import (
    AnnotationIndex,
    audit_annotations,
    decode_annotation_mask,
    group_records_by_filename,
    split_by_filename,
)
from solar_filament.data.dataset import SolarFilamentDataset, detection_collate_fn
from solar_filament.data.patches import (
    audit_patch_coverage,
    compute_patch_positions,
    extract_patch_targets,
    generate_patch_coords,
)

__all__ = [
    "AnnotationIndex",
    "SolarFilamentDataset",
    "audit_annotations",
    "audit_patch_coverage",
    "compute_patch_positions",
    "decode_annotation_mask",
    "detection_collate_fn",
    "extract_patch_targets",
    "generate_patch_coords",
    "group_records_by_filename",
    "split_by_filename",
]
