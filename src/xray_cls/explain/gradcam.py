import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_tensor, class_idx=0):
        self.model.zero_grad()
        output = self.model(input_tensor)

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlay