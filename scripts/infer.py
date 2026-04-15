import argparse
from pathlib import Path

import torch
from PIL import Image

from xray_cls.data.transforms import get_eval_transforms
from xray_cls.models.classifier import XRayClassifier
from xray_cls.utils.io import load_config


def main(config_path: str, checkpoint_path: str, image_path: str):
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
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    pred = 1 if prob >= 0.5 else 0
    label_name = "PNEUMONIA" if pred == 1 else "NORMAL"

    print(
        {
            "image_path": str(Path(image_path)),
            "probability": round(prob, 4),
            "prediction": label_name,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image_path)