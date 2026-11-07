
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from dataset import CLASS_NAMES


def build_model(pretrained=True):
    """ResNet18 with its last layer replaced by a 3-class layer."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def predict_probs(model, loader):
    """Return probabilities and true labels for every image."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            all_probs.append(torch.softmax(model(images), dim=1))
            all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_labels)
