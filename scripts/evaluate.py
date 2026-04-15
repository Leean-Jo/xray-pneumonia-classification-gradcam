import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xray_cls.data.dataset import XRayClassificationDataset
from xray_cls.data.transforms import get_eval_transforms
from xray_cls.engine.trainer import Trainer
from xray_cls.models.classifier import XRayClassifier
from xray_cls.utils.io import load_config


def main(config_path: str, checkpoint_path: str):
    config = load_config(config_path)
    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    test_dataset = XRayClassificationDataset(
        root_dir=config["data"]["root_dir"],
        split="test",
        transform=get_eval_transforms(config["data"]["image_size"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = XRayClassifier(
        model_name=config["model"]["name"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = Trainer(
        model=model,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=None,
        device=device,
    )

    metrics = trainer.validate(test_loader)
    print("Test Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()

    main(args.config, args.checkpoint)