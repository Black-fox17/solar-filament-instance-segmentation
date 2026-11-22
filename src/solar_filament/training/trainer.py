"""FP32 training and validation loop for Mask R-CNN solar filament segmentation."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from solar_filament.config import ProjectConfig
from solar_filament.utils.logging import get_logger

logger = get_logger(__name__)


def _move_batch_to_device(
    images: tuple[torch.Tensor, ...],
    targets: tuple[dict[str, Any], ...],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    non_blocking = device.type == "cuda"
    images_dev = [img.to(device, non_blocking=non_blocking) for img in images]
    targets_dev = [
        {
            k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v
            for k, v in target.items()
        }
        for target in targets
    ]
    return images_dev, targets_dev


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_clip_norm: float = 1.0,
    log_interval: int = 100,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Run one training epoch with non-finite loss/gradient guards and gradient clipping."""
    model.train()

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    scaler = torch.amp.GradScaler("cuda") if (use_amp and is_cuda) else None

    running_loss = 0.0
    component_sums: dict[str, float] = {}
    successful_batches = 0
    skipped_batches = 0
    total_samples = 0
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0

    start_time = time.time()
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=True)

    for batch_idx, (images, targets) in enumerate(progress):
        images_dev, targets_dev = _move_batch_to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)

        try:
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images_dev, targets_dev)
                    loss = sum(loss_dict.values())
            else:
                loss_dict = model(images_dev, targets_dev)
                loss = sum(loss_dict.values())
        except RuntimeError as exc:
            skipped_batches += 1
            logger.warning("Forward error at epoch %d batch %d: %s", epoch, batch_idx, exc)
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(loss):
            skipped_batches += 1
            logger.warning(
                "Non-finite loss at epoch %d batch %d: %s",
                epoch,
                batch_idx,
                {k: float(v.detach()) for k, v in loss_dict.items()},
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        bad_gradients = [
            name
            for name, param in model.named_parameters()
            if param.grad is not None and not torch.isfinite(param.grad).all()
        ]
        if bad_gradients:
            skipped_batches += 1
            logger.warning(
                "Non-finite gradients at epoch %d batch %d (first 5): %s",
                epoch,
                batch_idx,
                bad_gradients[:5],
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip_norm,
            error_if_nonfinite=True,
        )

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_loss = float(loss.detach().item())
        current_grad_norm = float(grad_norm.detach().item())
        running_loss += batch_loss
        successful_batches += 1
        total_samples += len(images_dev)
        gradient_norm_sum += current_grad_norm
        gradient_norm_max = max(gradient_norm_max, current_grad_norm)

        for name, value in loss_dict.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(value.detach().item())

        avg_loss = running_loss / successful_batches
        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{avg_loss:.4f}",
            grad=f"{current_grad_norm:.3f}",
            skip=skipped_batches,
        )

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            elapsed_min = (time.time() - start_time) / 60.0
            mem_info = (
                f" | GPU alloc={torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
                if is_cuda
                else ""
            )
            logger.info(
                "Epoch %d | Batch %d/%d | avg_loss=%.5f | grad=%.4f | lr=%.8f | elapsed=%.2fmin | skipped=%d%s",
                epoch,
                batch_idx + 1,
                len(loader),
                avg_loss,
                current_grad_norm,
                optimizer.param_groups[0]["lr"],
                elapsed_min,
                skipped_batches,
                mem_info,
            )

    if successful_batches == 0:
        raise RuntimeError(f"Epoch {epoch} failed: no successful training batches.")

    elapsed = time.time() - start_time
    peak_memory_gb = (torch.cuda.max_memory_allocated(device) / 1024**3) if is_cuda else 0.0

    return {
        "loss": running_loss / successful_batches,
        "components": {k: v / successful_batches for k, v in component_sums.items()},
        "successful_batches": successful_batches,
        "skipped_batches": skipped_batches,
        "gradient_norm_mean": gradient_norm_sum / successful_batches,
        "gradient_norm_max": gradient_norm_max,
        "duration_seconds": elapsed,
        "samples_per_second": total_samples / max(elapsed, 1e-8),
        "peak_memory_gb": peak_memory_gb,
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict[str, Any]:
    """Compute validation loss with BatchNorm running statistics frozen.

    Torchvision's Mask R-CNN returns loss dictionaries only when ``model.train()``
    is active. To avoid mutating BatchNorm running statistics during validation,
    all ``_BatchNorm`` submodules are temporarily set to ``eval()`` and restored
    in a ``finally`` block.
    """
    model.train()
    batchnorm_states: dict[nn.Module, bool] = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            batchnorm_states[module] = module.training
            module.eval()

    running_loss = 0.0
    component_sums: dict[str, float] = {}
    successful_batches = 0
    skipped_batches = 0

    start_time = time.time()
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=True)

    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(progress):
                images_dev, targets_dev = _move_batch_to_device(images, targets, device)

                try:
                    loss_dict = model(images_dev, targets_dev)
                    loss = sum(loss_dict.values())
                except RuntimeError as exc:
                    skipped_batches += 1
                    logger.warning(
                        "Validation error at epoch %d batch %d: %s", epoch, batch_idx, exc
                    )
                    continue

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    continue

                batch_loss = float(loss.item())
                running_loss += batch_loss
                successful_batches += 1

                for name, value in loss_dict.items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(value.item())

                progress.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    avg=f"{running_loss / successful_batches:.4f}",
                )
    finally:
        for module, was_training in batchnorm_states.items():
            module.train(was_training)
        model.train()

    if successful_batches == 0:
        raise RuntimeError("Validation produced no valid batches.")

    return {
        "loss": running_loss / successful_batches,
        "components": {k: v / successful_batches for k, v in component_sums.items()},
        "successful_batches": successful_batches,
        "skipped_batches": skipped_batches,
        "duration_seconds": time.time() - start_time,
    }


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_val_loss: float,
    history: list[dict[str, Any]],
    patience_counter: int,
) -> None:
    """Persist training state to a PyTorch checkpoint file."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "patience_counter": patience_counter,
        },
        ckpt_path,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ProjectConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Execute the full training loop with cosine LR decay, checkpointing, and early stopping."""
    tcfg = config.training
    checkpoint_dir = config.paths.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoint_dir / "best_model.pth"
    latest_ckpt = checkpoint_dir / "latest_model.pth"
    history_file = checkpoint_dir / "training_history.json"

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tcfg.epochs,
        eta_min=tcfg.eta_min,
    )

    if tcfg.use_amp:
        logger.warning(
            "use_amp=True requested. Note that FP16 autocast caused non-finite gradients "
            "during the original A100 Mask R-CNN experiments; FP32 (use_amp=False) is recommended."
        )

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []
    training_start = time.time()

    logger.info(
        "Starting training on %s | train_patches=%d | val_patches=%d | batch_size=%d | epochs=%d | amp=%s",
        device,
        len(train_loader.dataset),
        len(val_loader.dataset),
        tcfg.batch_size,
        tcfg.epochs,
        tcfg.use_amp,
    )

    for epoch in range(1, tcfg.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=tcfg.epochs,
            grad_clip_norm=tcfg.grad_clip_norm,
            log_interval=tcfg.log_interval,
            use_amp=tcfg.use_amp,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=tcfg.epochs,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        improved = val_metrics["loss"] < best_val_loss - tcfg.min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                best_ckpt,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                history,
                patience_counter,
            )
            logger.info("Saved new best checkpoint at epoch %d (val_loss=%.5f)", epoch, best_val_loss)
        else:
            patience_counter += 1

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_components": train_metrics["components"],
            "val_components": val_metrics["components"],
            "train_batches": train_metrics["successful_batches"],
            "skipped_train_batches": train_metrics["skipped_batches"],
            "skipped_val_batches": val_metrics["skipped_batches"],
            "gradient_norm_mean": train_metrics["gradient_norm_mean"],
            "gradient_norm_max": train_metrics["gradient_norm_max"],
            "train_duration_minutes": train_metrics["duration_seconds"] / 60.0,
            "val_duration_minutes": val_metrics["duration_seconds"] / 60.0,
            "epoch_duration_minutes": epoch_duration / 60.0,
            "samples_per_second": train_metrics["samples_per_second"],
            "peak_gpu_memory_gb": train_metrics["peak_memory_gb"],
            "best_val_loss": best_val_loss,
        }
        history.append(epoch_record)

        save_checkpoint(
            latest_ckpt,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val_loss,
            history,
            patience_counter,
        )
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

        logger.info(
            "Epoch %d/%d complete | train_loss=%.5f | val_loss=%.5f | best_val=%.5f | patience=%d/%d",
            epoch,
            tcfg.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            best_val_loss,
            patience_counter,
            tcfg.patience,
        )

        if train_metrics["skipped_batches"] > len(train_loader) * tcfg.max_skipped_batch_ratio:
            raise RuntimeError(
                f"More than {tcfg.max_skipped_batch_ratio * 100:.1f}% of training batches were skipped "
                f"({train_metrics['skipped_batches']}/{len(train_loader)}). Stopping to protect stability."
            )

        if patience_counter >= tcfg.patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", tcfg.patience)
            break

        gc.collect()

    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        logger.info(
            "Restored best model from epoch %s (val_loss=%.5f)",
            best_state.get("epoch"),
            best_state.get("best_val_loss", float("nan")),
        )

    logger.info(
        "Training completed in %.2f hours | epochs=%d | best_val_loss=%.5f",
        (time.time() - training_start) / 3600.0,
        len(history),
        best_val_loss,
    )
    return history
