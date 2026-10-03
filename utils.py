from contextlib import nullcontext
from pathlib import Path
import json
import logging
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def setup_logger(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cnn_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def separator(char="=", width=72):
    return char * width


def get_autocast_context(device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.amp.GradScaler("cuda")
    return None


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()

    loss_sum = 0.0
    targets_all = []
    preds_all = []
    probs_all = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with get_autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        loss_sum += loss.item() * images.size(0)
        targets_all.append(targets.cpu().numpy())
        preds_all.append(predictions.cpu().numpy())
        probs_all.append(probabilities.cpu().numpy())

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(preds_all)
    probabilities = np.concatenate(probs_all)

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    top_k = min(5, probabilities.shape[1])
    topk_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top5 = np.mean([target in row for target, row in zip(targets, topk_indices)])

    return {
        "loss": float(loss_sum / len(loader.dataset)),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top5_accuracy": float(top5),
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def save_history_plots(history_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="Validation macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Validation macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_curve.png", dpi=170)
    plt.close()


def save_confusion_matrix(targets, predictions, class_names, output_path):
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_per_class_metrics(targets, predictions, class_names, output_path):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    df = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    df.to_csv(output_path, index=False)
    return df


def benchmark_latency(model, device, image_size=224, warmup=20, iterations=100, use_amp=True):
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            with get_autocast_context(device, use_amp):
                _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with get_autocast_context(device, use_amp):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    latency_ms = float(np.median(timings))
    fps = 1000.0 / latency_ms
    return latency_ms, fps


def save_json(data, path):
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=convert)


def save_summary_plots(summary_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("test_accuracy", "Test accuracy", "comparison_accuracy.png"),
        ("test_macro_f1", "Test macro-F1", "comparison_macro_f1.png"),
        ("params_million", "Parameters (million)", "comparison_parameters.png"),
        ("model_size_mb", "Checkpoint size (MB)", "comparison_model_size.png"),
        ("latency_ms", "Median latency (ms / image)", "comparison_latency.png"),
    ]

    for column, title, filename in metrics:
        plt.figure(figsize=(9, 5))
        plt.bar(summary_df["display_name"], summary_df[column])
        plt.title(title)
        plt.ylabel(title)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=170)
        plt.close()
