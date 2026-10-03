import torch.nn as nn
from torchvision import models


MODEL_NAMES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large",
]


def build_model(model_name, num_classes, pretrained=True):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")

    return model


def set_classifier_trainable(model_name, model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    module = model.fc if model_name.startswith("resnet") else model.classifier
    for parameter in module.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
