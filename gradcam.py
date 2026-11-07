
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # ResNet18's final convolutional block
        self.target_layer = self.model.layer4[-1]
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output

        # Save gradients only when gradient tracking is enabled.
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def __call__(self, image_tensor, target_class):
        self.gradients = None
        self.activations = None

        # Grad-CAM needs gradients, even if called from a no_grad context.
        with torch.enable_grad():
            image_tensor = image_tensor.detach().clone().requires_grad_(True)

            output = self.model(image_tensor.unsqueeze(0))
            score = output[0, target_class]

            self.model.zero_grad(set_to_none=True)
            score.backward()

            if self.activations is None or self.gradients is None:
                raise RuntimeError(
                    "Grad-CAM could not collect activations or gradients."
                )

            activations = self.activations.detach()
            gradients = self.gradients.detach()

            # Average gradients across the feature-map dimensions.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted sum of feature maps.
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize the heatmap to the input image size.
            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0]

            # Normalize values to the range 0 to 1.
            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            cam = cam.detach().cpu().numpy()

        return cam, activations, gradients
