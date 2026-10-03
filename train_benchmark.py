from pathlib import Path
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import ExperimentConfig
from data import build_dataloaders
from models import MODEL_NAMES, build_model, count_parameters, set_classifier_trainable, unfreeze_all
from utils import (
    benchmark_latency,
    evaluate,
    get_autocast_context,
    make_grad_scaler,
    save_confusion_matrix,
    save_history_plots,
    save_json,
    save_per_class_metrics,
    save_summary_plots,
    separator,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet / EfficientNet / MobileNet benchmark")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--include-scratch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--scratch-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += batch_size

    return {"loss": running_loss / total, "accuracy": correct / total}


def run_training_phase(
    *, phase_name, model, train_loader, val_loader, optimizer, scheduler,
    criterion, device, epochs, start_epoch, logger, use_amp, best_state,
    best_val_f1, patience, early_stopping_patience,
):
    scaler = make_grad_scaler(device, use_amp)
    history_rows = []

    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

                epoch_seconds = time.perf_counter() - epoch_start

        # Bu epoch boyunca kullanÄ±lan learning rate
        lr = optimizer.param_groups[0]["lr"]

        # GPU peak memory
        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        # Scheduler bir sonraki epoch iÃ§in LR'yi gÃ¼nceller
        if scheduler is not None:
            scheduler.step()

        improved = val_metrics["macro_f1"] > best_val_f1

        if improved:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience = 0
        else:
            patience += 1

        row = {
            "epoch": global_epoch,
            "phase": phase_name,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "learning_rate": lr,
            "epoch_seconds": epoch_seconds,
            "gpu_peak_memory_gb": peak_memory_gb,
        }

        logger.info("")
        logger.info(f"Epoch {global_epoch:02d} | {phase_name}")
        logger.info("-" * 72)
        logger.info(f"Train Loss       : {train_metrics['loss']:.5f}")
        logger.info(f"Train Accuracy   : {train_metrics['accuracy'] * 100:6.2f} %")
        logger.info(f"Val Loss         : {val_metrics['loss']:.5f}")
        logger.info(f"Val Accuracy     : {val_metrics['accuracy'] * 100:6.2f} %")
        logger.info(f"Val Macro-F1     : {val_metrics['macro_f1']:.5f}")
        logger.info(f"Val Top-5 Acc    : {val_metrics['top5_accuracy'] * 100:6.2f} %")
        logger.info(f"Learning Rate    : {lr:.7f}")
        logger.info(f"Epoch Time       : {epoch_seconds:.2f} s")
        if device.type == "cuda":
            logger.info(f"GPU Peak Memory  : {peak_memory_gb:.2f} GB")
        if improved:
            logger.info("âœ“ New best checkpoint (validation Macro-F1)")

        if early_stopping_patience > 0 and patience >= early_stopping_patience:
            logger.info(f"Early stopping: {patience} epoch boyunca iyileÅŸme yok.")
            break

    return history_rows, best_state, best_val_f1, patience, start_epoch + len(history_rows)


def train_and_evaluate_model(
    *, model_name, display_name, pretrained, config, loaders, device,
    logger, scratch=False,
):
    model_dir = Path(config.output_root) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    classes = loaders["classes"]
    num_classes = len(classes)
    set_seed(config.seed)

    logger.info("")
    logger.info(separator())
    logger.info(f"MODEL: {display_name}")
    logger.info(separator())

    base_name = model_name.replace("_scratch", "")
    model = build_model(base_name, num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    total_params, _ = count_parameters(model)

    logger.info(f"Pretrained       : {pretrained}")
    logger.info(f"Total Parameters : {total_params:,}")
    logger.info(f"Classes          : {num_classes}")

    history = []
    best_state = None
    best_val_f1 = -1.0
    patience = 0
    current_epoch = 0
    training_start = time.perf_counter()

    if scratch:
        unfreeze_all(model)
        _, trainable_params = count_parameters(model)
        logger.info(f"Trainable Params : {trainable_params:,}")
        logger.info("Training mode    : From scratch")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.scratch_lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.scratch_epochs)
        )

        rows, best_state, best_val_f1, patience, current_epoch = run_training_phase(
            phase_name="scratch", model=model, train_loader=loaders["train"],
            val_loader=loaders["val"], optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, device=device, epochs=config.scratch_epochs,
            start_epoch=current_epoch, logger=logger, use_amp=config.use_amp,
            best_state=best_state, best_val_f1=best_val_f1, patience=patience,
            early_stopping_patience=config.early_stopping_patience,
        )
        history.extend(rows)

    else:
        set_classifier_trainable(base_name, model)
        _, trainable_params = count_parameters(model)
        logger.info(f"Warm-up trainable: {trainable_params:,}")
        logger.info(f"Phase 1          : classifier warm-up ({config.warmup_epochs} epoch)")
        logger.info(f"Phase 2          : full fine-tuning ({config.finetune_epochs} epoch)")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.warmup_lr,
            weight_decay=config.weight_decay,
        )
        rows, best_state, best_val_f1, patience, current_epoch = run_training_phase(
            phase_name="warmup", model=model, train_loader=loaders["train"],
            val_loader=loaders["val"], optimizer=optimizer, scheduler=None,
            criterion=criterion, device=device, epochs=config.warmup_epochs,
            start_epoch=current_epoch, logger=logger, use_amp=config.use_amp,
            best_state=best_state, best_val_f1=best_val_f1, patience=patience,
            early_stopping_patience=0,
        )
        history.extend(rows)

        unfreeze_all(model)
        _, trainable_params = count_parameters(model)
        logger.info("")
        logger.info(f"Full fine-tune trainable params: {trainable_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.finetune_epochs)
        )
        patience = 0
        rows, best_state, best_val_f1, patience, current_epoch = run_training_phase(
            phase_name="finetune", model=model, train_loader=loaders["train"],
            val_loader=loaders["val"], optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, device=device, epochs=config.finetune_epochs,
            start_epoch=current_epoch, logger=logger, use_amp=config.use_amp,
            best_state=best_state, best_val_f1=best_val_f1, patience=patience,
            early_stopping_patience=config.early_stopping_patience,
        )
        history.extend(rows)

    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    checkpoint_path = model_dir / "best_model.pth"
    torch.save({
        "model_name": model_name,
        "display_name": display_name,
        "pretrained": pretrained,
        "state_dict": model.state_dict(),
        "classes": classes,
        "config": config.to_dict(),
        "best_val_macro_f1": best_val_f1,
    }, checkpoint_path)
    model_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / "history.csv", index=False)
    save_history_plots(history_df, model_dir)

    logger.info("")
    logger.info("TEST EVALUATION")
    logger.info("-" * 72)

    test_metrics = evaluate(model, loaders["test"], criterion, device, config.use_amp)
    latency_ms, fps = benchmark_latency(
        model, device, image_size=config.image_size, use_amp=config.use_amp
    )

    save_confusion_matrix(
        test_metrics["targets"], test_metrics["predictions"], classes,
        model_dir / "confusion_matrix.png"
    )
    save_per_class_metrics(
        test_metrics["targets"], test_metrics["predictions"], classes,
        model_dir / "per_class_metrics.csv"
    )

    np.save(model_dir / "test_targets.npy", test_metrics["targets"])
    np.save(model_dir / "test_predictions.npy", test_metrics["predictions"])
    np.save(model_dir / "test_probabilities.npy", test_metrics["probabilities"])

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )

    result = {
        "model_name": model_name,
        "display_name": display_name,
        "pretrained": pretrained,
        "total_parameters": total_params,
        "params_million": total_params / 1e6,
        "best_val_macro_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_top5_accuracy": test_metrics["top5_accuracy"],
        "training_seconds": training_seconds,
        "model_size_mb": model_size_mb,
        "latency_ms": latency_ms,
        "fps": fps,
        "gpu_peak_memory_gb": peak_memory_gb,
    }
    save_json(result, model_dir / "final_metrics.json")

    logger.info(f"Test Loss        : {result['test_loss']:.5f}")
    logger.info(f"Test Accuracy    : {result['test_accuracy'] * 100:6.2f} %")
    logger.info(f"Macro Precision  : {result['test_macro_precision']:.5f}")
    logger.info(f"Macro Recall     : {result['test_macro_recall']:.5f}")
    logger.info(f"Macro F1         : {result['test_macro_f1']:.5f}")
    logger.info(f"Top-5 Accuracy   : {result['test_top5_accuracy'] * 100:6.2f} %")
    logger.info(f"Training Time    : {training_seconds / 60:.2f} min")
    logger.info(f"Checkpoint Size  : {model_size_mb:.2f} MB")
    logger.info(f"Latency (median) : {latency_ms:.3f} ms / image")
    logger.info(f"Estimated FPS    : {fps:.2f}")
    if device.type == "cuda":
        logger.info(f"GPU Peak Memory  : {peak_memory_gb:.2f} GB")

    return result


def write_text_report(summary_df, config, loaders, path):
    lines = [
        "CNN ARCHITECTURE BENCHMARK REPORT",
        "=" * 72,
        "",
        "DATASET",
        "-" * 72,
        "Oxford-IIIT Pet",
        f"Classes : {len(loaders['classes'])}",
        f"Train   : {loaders['sizes']['train']}",
        f"Val     : {loaders['sizes']['val']}",
        f"Test    : {loaders['sizes']['test']}",
        f"Image   : {config.image_size} x {config.image_size}",
        "",
        "FINAL RESULTS",
        "-" * 72,
    ]

    printable = summary_df[[
        "display_name", "test_accuracy", "test_macro_f1", "params_million",
        "model_size_mb", "latency_ms", "fps", "training_seconds"
    ]].copy()
    lines.append(printable.to_string(index=False))
    lines += [
        "",
        "INTERPRETATION GUIDE",
        "-" * 72,
        "- Accuracy / Macro-F1: classification quality.",
        "- Params / checkpoint MB: model capacity and storage cost.",
        "- Latency / FPS: single-image inference cost on this machine.",
        "- Training time: hardware- and environment-dependent.",
        "- One model need not dominate every metric.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        data_root=args.data_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        scratch_epochs=args.scratch_epochs,
        use_amp=not args.no_amp,
    )

    output_root = Path(config.output_root)
    summary_dir = output_root / "summary"
    logs_dir = output_root / "logs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(logs_dir / "experiment.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA bulunamadÄ±. Ã–nce `python check_env.py` Ã§alÄ±ÅŸtÄ±r ve CUDA destekli "
            "PyTorch/Torchvision kurulumunu dÃ¼zelt."
        )

    logger.info(separator())
    logger.info("CNN ARCHITECTURE BENCHMARK")
    logger.info("ResNet18 | ResNet50 | EfficientNet-B0 | MobileNetV3-Large")
    logger.info(separator())
    logger.info("")
    logger.info("DEVICE")
    logger.info("-" * 72)
    logger.info(f"PyTorch       : {torch.__version__}")
    logger.info(f"CUDA build    : {torch.version.cuda}")
    logger.info(f"Device        : {device}")
    if device.type == "cuda":
        logger.info(f"GPU           : {torch.cuda.get_device_name(0)}")
        logger.info(f"AMP           : {'Enabled' if config.use_amp else 'Disabled'}")
    else:
        logger.info("WARNING       : CUDA yok. EÄŸitim CPU'da Ã§alÄ±ÅŸacak.")

    logger.info("")
    logger.info("DATASET PREPARATION")
    logger.info("-" * 72)
    logger.info("Oxford-IIIT Pet indiriliyor / doÄŸrulanÄ±yor...")
    loaders = build_dataloaders(
        config.data_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    logger.info("Dataset        : Oxford-IIIT Pet")
    logger.info(f"Classes        : {len(loaders['classes'])}")
    logger.info(f"Image size     : {config.image_size} x {config.image_size}")
    logger.info(f"Train          : {loaders['sizes']['train']}")
    logger.info(f"Validation     : {loaders['sizes']['val']}")
    logger.info(f"Test           : {loaders['sizes']['test']}")
    logger.info(f"Batch size     : {config.batch_size}")
    logger.info(f"Seed           : {config.seed}")

    save_json({
        "config": config.to_dict(),
        "dataset": {
            "name": "Oxford-IIIT Pet",
            "classes": loaders["classes"],
            "sizes": loaders["sizes"],
        },
        "device": {
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }, summary_dir / "config_snapshot.json")

    display_names = {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v3_large": "MobileNetV3-Large",
    }

    results = []
    for model_name in args.models:
        results.append(train_and_evaluate_model(
            model_name=model_name,
            display_name=display_names[model_name],
            pretrained=True,
            config=config,
            loaders=loaders,
            device=device,
            logger=logger,
            scratch=False,
        ))

    if args.include_scratch:
        results.append(train_and_evaluate_model(
            model_name="resnet18_scratch",
            display_name="ResNet18-Scratch",
            pretrained=False,
            config=config,
            loaders=loaders,
            device=device,
            logger=logger,
            scratch=True,
        ))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_dir / "model_comparison.csv", index=False)
    save_json(results, summary_dir / "model_comparison.json")
    save_summary_plots(summary_df, summary_dir)
    write_text_report(summary_df, config, loaders, summary_dir / "experiment_report.txt")

    logger.info("")
    logger.info(separator())
    logger.info("FINAL TEST RESULTS")
    logger.info(separator())
    columns = [
        "display_name", "test_accuracy", "test_macro_f1", "params_million",
        "model_size_mb", "latency_ms", "fps"
    ]
    logger.info(summary_df[columns].to_string(index=False))
    logger.info("")
    logger.info("Kaydedilen ana Ã§Ä±ktÄ±lar:")
    logger.info(f"- {summary_dir / 'model_comparison.csv'}")
    logger.info(f"- {summary_dir / 'model_comparison.json'}")
    logger.info(f"- {summary_dir / 'experiment_report.txt'}")
    logger.info(f"- {logs_dir / 'experiment.log'}")
    logger.info("- Her model klasÃ¶rÃ¼nde checkpoint, history, curves, confusion matrix ve per-class metrics bulunur.")
    logger.info(separator())


if __name__ == "__main__":
    main()
