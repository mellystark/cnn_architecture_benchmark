# CNN Architecture Benchmark

PyTorch ile aynı görüntü sınıflandırma problemi üzerinde farklı CNN mimarilerini karşılaştıran yerel laboratuvar projesi.

## Araştırma sorusu

Aynı veri ve eğitim protokolü altında mimari seçimi:

- sınıflandırma doğruluğunu,
- Macro-F1 skorunu,
- parametre sayısını,
- checkpoint boyutunu,
- inference gecikmesini,
- FPS değerini,
- eğitim süresini

nasıl etkiliyor?

## Dataset

**Oxford-IIIT Pet**

- 37 kedi/köpek ırkı
- Torchvision üzerinden otomatik indirme
- `trainval` split'i stratified olarak train/validation'a ayrılır
- resmi `test` split'i final test için kullanılır
- giriş boyutu: `3 x 224 x 224`

## Modeller

Ana benchmark:

1. ResNet18
2. ResNet50
3. EfficientNet-B0
4. MobileNetV3-Large

Opsiyonel ablation:

5. ResNet18-Scratch

Ana modeller ImageNet pretrained ağırlıklarıyla başlatılır.

## Eğitim protokolü

Pretrained modeller:

```text
Phase 1
Backbone frozen
Classifier trainable
3 epoch warm-up

        ↓

Phase 2
All layers trainable
10 epoch fine-tuning
```

Varsayılan optimizer:

```text
AdamW
```

Loss:

```text
CrossEntropyLoss
```

Fine-tuning scheduler:

```text
CosineAnnealingLR
```

CUDA varsa mixed precision (AMP) otomatik kullanılır.

## 1. Ortam kontrolü

Önce Miniconda ortamını aç:

```powershell
conda activate llm_ders
```

Sonra:

```powershell
python check_env.py
```

İdeal çıktı:

```text
CUDA ready   : True
GPU          : NVIDIA GeForce RTX 3060 Laptop GPU
```

`CUDA ready: False` görürsen benchmark'a başlamadan önce CUDA destekli PyTorch ve Torchvision kurulmalıdır.

## 2. Genel Python paketleri

```powershell
pip install -r requirements.txt
```

> `torch` ve `torchvision` requirements dosyasına özellikle eklenmedi. CPU / CUDA paketi donanıma göre farklı seçildiği için PyTorch'un resmi kurulum seçicisinden uygun CUDA build kurulmalıdır.

## 3. Hızlı test

Önce yalnız ResNet18 ile birer epoch çalıştır:

```powershell
python train_benchmark.py --models resnet18 --warmup-epochs 1 --finetune-epochs 1 --require-cuda
```

Bu adım:

- dataset indirmesini,
- GPU erişimini,
- model indirmesini,
- output klasörlerini

kontrol etmek için kullanılır.

## 4. Ana benchmark

```powershell
python train_benchmark.py --require-cuda
```

Varsayılan modeller:

```text
ResNet18
ResNet50
EfficientNet-B0
MobileNetV3-Large
```

## 5. Transfer learning ablation

Ana benchmark + ResNet18 scratch:

```powershell
python train_benchmark.py --include-scratch --require-cuda
```

## Çıktılar

```text
outputs/
├── logs/
│   └── experiment.log
│
├── summary/
│   ├── config_snapshot.json
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   ├── experiment_report.txt
│   ├── comparison_accuracy.png
│   ├── comparison_macro_f1.png
│   ├── comparison_parameters.png
│   ├── comparison_model_size.png
│   └── comparison_latency.png
│
├── resnet18/
│   ├── best_model.pth
│   ├── history.csv
│   ├── final_metrics.json
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── macro_f1_curve.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.csv
│   ├── test_targets.npy
│   ├── test_predictions.npy
│   └── test_probabilities.npy
│
├── resnet50/
├── efficientnet_b0/
└── mobilenet_v3_large/
```

## Terminalde raporlanan metrikler

Her epoch:

- Train Loss
- Train Accuracy
- Validation Loss
- Validation Accuracy
- Validation Macro-F1
- Validation Top-5 Accuracy
- Learning rate
- Epoch süresi
- GPU peak memory

Final test:

- Test Loss
- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Top-5 Accuracy
- Toplam eğitim süresi
- Checkpoint boyutu
- Median single-image latency
- Tahmini FPS
- GPU peak memory

## Neden bu modeller?

### ResNet18
Residual bağlantılar için hafif ve güçlü baseline.

### ResNet50
Aynı ailede derinlik / kapasite artışının maliyet-fayda karşılaştırması.

### EfficientNet-B0
Depth, width ve resolution'ın dengeli ölçeklenmesi fikrini temsil eder.

### MobileNetV3-Large
Edge ve mobil dağıtım için hafif mimari yaklaşımını temsil eder.

### ResNet18-Scratch
Transfer learning'in etkisini doğrudan ölçmek için kullanılır.

## Deney sonunda cevaplanacak sorular

1. Daha derin ResNet50, ResNet18'e karşı ne kadar kazanç sağlıyor?
2. EfficientNet-B0 accuracy / parametre / latency dengesinde nasıl davranıyor?
3. MobileNetV3-Large daha düşük hesaplama maliyetinde ne kadar başarı koruyor?
4. ImageNet pretrained ResNet18, scratch ResNet18'e ne kadar avantaj sağlıyor?
5. En yüksek accuracy ile en düşük latency aynı modelde mi?
