import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from xray_cls.data.transforms import get_eval_transforms
from xray_cls.explain.gradcam import GradCAM, overlay_cam_on_image
from xray_cls.models.classifier import XRayClassifier
from xray_cls.utils.io import ensure_dir, load_config


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def main(config_path: str, checkpoint_path: str, image_path: str, save_dir: str):
    config = load_config(config_path)
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    model = XRayClassifier(
        model_name=config["model"]["name"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("L")
    transform = get_eval_transforms(config["data"]["image_size"])
    input_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, target_layer=model.backbone.layer4[-1])
    cam = gradcam.generate(input_tensor, class_idx=0)

    base_image = denormalize(input_tensor[0])
    overlay = overlay_cam_on_image(base_image, cam)

    ensure_dir(save_dir)
    stem = Path(image_path).stem
    save_path = Path(save_dir) / f"{stem}_gradcam.png"

    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM to {save_path}")

    gradcam.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/figures")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image_path, args.save_dir)