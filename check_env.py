import platform
import sys

import torch
import torchvision


def line():
    print("=" * 72)


line()
print("CNN BENCHMARK - ENVIRONMENT CHECK")
line()
print(f"Python       : {sys.version.split()[0]}")
print(f"Platform     : {platform.platform()}")
print(f"PyTorch      : {torch.__version__}")
print(f"Torchvision  : {torchvision.__version__}")
print(f"CUDA build   : {torch.version.cuda}")
print(f"CUDA ready   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM         : {props.total_memory / (1024**3):.2f} GB")
    print(f"cuDNN        : {torch.backends.cudnn.version()}")
    print()
    print("OK: CUDA destekli PyTorch aktif.")
else:
    print("GPU          : Kullanılmıyor")
    print()
    print("UYARI:")
    print("Bu ortamda PyTorch CUDA'ya erişemiyor.")
    print("Benchmark CPU'da çalışır ama özellikle ResNet50 için yavaş olur.")
    print("CUDA destekli PyTorch/Torchvision kurulumunu düzeltip tekrar kontrol et.")

line()
