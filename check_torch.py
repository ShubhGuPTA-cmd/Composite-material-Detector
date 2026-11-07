
import torch
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU available:", torch.cuda.is_available())

# Create one fake colour image of size 224 x 224
x = torch.rand(1, 3, 224, 224)

# Load a pretrained image recognition model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()

# Test the model
with torch.no_grad():
    out = model(x)

print("model output shape:", out.shape)
print("OK: PyTorch and the pretrained model work.")
