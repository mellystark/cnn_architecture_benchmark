
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET = "Oxford-IIIT Pet"
GPU = "NVIDIA GeForce RTX 3060 Laptop GPU"


def ensure(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def finish(fig, out_path: Path, exp_title: str, plot_title: str):
    fig.suptitle(f"{exp_title}\n{plot_title}", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.015,
        f"Dataset: {DATASET} | 37 sınıf | 224x224 | GPU: {GPU}",
        ha="center", fontsize=8
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.90])
    ensure(out_path.parent)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def bar(df, col, ylabel, title, out_path, exp_title, scale=1.0):
    vals = df[col] * scale
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["display_name"], vals)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width()/2,
            b.get_height(),
            f"{v:.2f}",
            ha="center", va="bottom", fontsize=9
        )

    finish(fig, out_path, exp_title, title)


def scatter(df, xcol, ycol, xlabel, ylabel, title, out_path, exp_title, yscale=1.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy() * yscale
    ax.scatter(x, y, s=90)

    for xi, yi, name in zip(x, y, df["display_name"]):
        ax.annotate(name, (xi, yi), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    finish(fig, out_path, exp_title, title)


def read_classes(root: Path):
    p = root / "summary" / "config_snapshot.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            classes = data.get("dataset", {}).get("classes", [])
            if classes:
                return classes
        except Exception:
            pass
    return [f"class_{i}" for i in range(37)]


def display_name_from_folder(name: str):
    return {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "efficientnet_b0": "EfficientNet-B0",
        "mobilenet_v3_large": "MobileNetV3-Large",
        "resnet18_scratch": "ResNet18-Scratch",
    }.get(name, name)


def history_plot(root: Path, folders, metric, ylabel, title, out_path, exp_title, scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    found = False

    for folder in folders:
        p = root / folder / "history.csv"
        if not p.exists():
            continue

        df = pd.read_csv(p)
        if metric not in df.columns:
            continue

        found = True
        ax.plot(
            df["epoch"],
            df[metric] * scale,
            marker="o",
            markersize=3,
            label=display_name_from_folder(folder)
        )

    if not found:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, out_path, exp_title, title)


def per_model_plots(root: Path, folder: str, exp_title: str, out_dir: Path):
    p = root / folder / "history.csv"
    if not p.exists():
        return

    df = pd.read_csv(p)
    name = display_name_from_folder(folder)
    model_out = ensure(out_dir / "per_model" / folder)

    # Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, model_out / "01_loss.png", exp_title, f"{name} - Loss Eğrileri")

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_accuracy"] * 100, marker="o", label="Train Accuracy")
    ax.plot(df["epoch"], df["val_accuracy"] * 100, marker="o", label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, model_out / "02_accuracy.png", exp_title, f"{name} - Accuracy Eğrileri")

    # Macro-F1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["val_macro_f1"], marker="o", label="Validation Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, model_out / "03_macro_f1.png", exp_title, f"{name} - Validation Macro-F1")

    # LR
    if "learning_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["learning_rate"], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(alpha=0.25)
        finish(fig, model_out / "04_learning_rate.png", exp_title, f"{name} - Learning Rate")

    # GPU memory
    if "gpu_peak_memory_gb" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df["gpu_peak_memory_gb"], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Peak GPU Memory (GB)")
        ax.grid(alpha=0.25)
        finish(fig, model_out / "05_gpu_memory.png", exp_title, f"{name} - GPU Bellek Kullanımı")

    # Confusion matrix from saved arrays
    tpath = root / folder / "test_targets.npy"
    ppath = root / folder / "test_predictions.npy"
    if tpath.exists() and ppath.exists():
        targets = np.load(tpath)
        preds = np.load(ppath)
        classes = read_classes(root)
        n = len(classes)

        cm = np.zeros((n, n), dtype=int)
        for t, p_ in zip(targets, preds):
            cm[int(t), int(p_)] += 1

        fig, ax = plt.subplots(figsize=(15, 13))
        im = ax.imshow(cm)
        ax.set_xlabel("Tahmin edilen sınıf")
        ax.set_ylabel("Gerçek sınıf")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        fig.colorbar(im, ax=ax)
        finish(fig, model_out / "06_confusion_matrix.png", exp_title, f"{name} - Test Confusion Matrix")


def experiment_1(root: Path):
    summary = root / "summary" / "model_comparison.csv"
    if not summary.exists():
        print(f"[ATLANDI] Deney 1 bulunamadı: {summary}")
        return

    exp = "DENEY 1 - CNN MİMARİ KARŞILAŞTIRMASI"
    out = ensure(root / "figures_labeled" / "experiment_1_architecture_benchmark")
    df = pd.read_csv(summary)

    bar(df, "test_accuracy", "Test Accuracy (%)", "Test Accuracy Karşılaştırması", out/"01_test_accuracy.png", exp, 100)
    bar(df, "test_macro_f1", "Macro-F1", "Macro-F1 Karşılaştırması", out/"02_macro_f1.png", exp)
    bar(df, "params_million", "Parametre (Milyon)", "Model Parametre Sayıları", out/"03_parameters.png", exp)
    bar(df, "model_size_mb", "Checkpoint Boyutu (MB)", "Model Boyutu Karşılaştırması", out/"04_model_size.png", exp)
    bar(df, "latency_ms", "Latency (ms/görüntü)", "Inference Latency Karşılaştırması", out/"05_latency.png", exp)
    bar(df, "fps", "FPS", "Inference FPS Karşılaştırması", out/"06_fps.png", exp)

    if "training_seconds" in df.columns:
        temp = df.copy()
        temp["training_minutes"] = temp["training_seconds"] / 60
        bar(temp, "training_minutes", "Dakika", "Toplam Eğitim Süresi", out/"07_training_time.png", exp)

    scatter(df, "latency_ms", "test_accuracy", "Latency (ms/görüntü)", "Test Accuracy (%)",
            "Accuracy - Latency Dengesi", out/"08_accuracy_vs_latency.png", exp, 100)

    scatter(df, "params_million", "test_accuracy", "Parametre (Milyon)", "Test Accuracy (%)",
            "Accuracy - Parametre Dengesi", out/"09_accuracy_vs_parameters.png", exp, 100)

    folders = ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"]

    history_plot(root, folders, "val_accuracy", "Validation Accuracy (%)",
                 "Modellerin Validation Accuracy Eğrileri", out/"10_val_accuracy_by_epoch.png", exp, 100)

    history_plot(root, folders, "val_macro_f1", "Validation Macro-F1",
                 "Modellerin Validation Macro-F1 Eğrileri", out/"11_val_macro_f1_by_epoch.png", exp)

    for folder in folders:
        per_model_plots(root, folder, exp, out)

    print(f"[OK] Deney 1 grafikleri: {out}")


def experiment_2(root: Path):
    summary = root / "summary" / "model_comparison.csv"
    if not summary.exists():
        print(f"[ATLANDI] Deney 2 bulunamadı: {summary}")
        return

    exp = "DENEY 2 - TRANSFER LEARNING ABLATION"
    out = ensure(root / "figures_labeled" / "experiment_2_transfer_learning")
    df = pd.read_csv(summary)

    bar(df, "test_accuracy", "Test Accuracy (%)", "Pretrained vs Scratch - Test Accuracy", out/"01_test_accuracy.png", exp, 100)
    bar(df, "test_macro_f1", "Macro-F1", "Pretrained vs Scratch - Macro-F1", out/"02_macro_f1.png", exp)

    if "test_top5_accuracy" in df.columns:
        bar(df, "test_top5_accuracy", "Top-5 Accuracy (%)", "Pretrained vs Scratch - Top-5 Accuracy", out/"03_top5_accuracy.png", exp, 100)

    bar(df, "model_size_mb", "Checkpoint Boyutu (MB)", "Model Boyutu Karşılaştırması", out/"04_model_size.png", exp)
    bar(df, "latency_ms", "Latency (ms/görüntü)", "Inference Latency Karşılaştırması", out/"05_latency.png", exp)
    bar(df, "fps", "FPS", "Inference FPS Karşılaştırması", out/"06_fps.png", exp)

    folders = ["resnet18", "resnet18_scratch"]

    history_plot(root, folders, "val_accuracy", "Validation Accuracy (%)",
                 "Pretrained vs Scratch - Validation Accuracy", out/"07_val_accuracy_by_epoch.png", exp, 100)

    history_plot(root, folders, "val_macro_f1", "Validation Macro-F1",
                 "Pretrained vs Scratch - Validation Macro-F1", out/"08_val_macro_f1_by_epoch.png", exp)

    history_plot(root, folders, "train_accuracy", "Train Accuracy (%)",
                 "Pretrained vs Scratch - Train Accuracy", out/"09_train_accuracy_by_epoch.png", exp, 100)

    for folder in folders:
        per_model_plots(root, folder, exp, out)

    print(f"[OK] Deney 2 grafikleri: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="outputs")
    parser.add_argument("--ablation-root", default="outputs_transfer_ablation")
    args = parser.parse_args()

    print("=" * 72)
    print("DENEY ETİKETLİ GRAFİK ÜRETİCİ")
    print("=" * 72)

    experiment_1(Path(args.benchmark_root))
    experiment_2(Path(args.ablation_root))

    print("=" * 72)
    print("Tamamlandı.")
    print("=" * 72)


if __name__ == "__main__":
    main()
