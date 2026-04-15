
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from xray_cls.data.dataset import XRayClassificationDataset
from xray_cls.data.transforms import get_eval_transforms, get_train_transforms
from xray_cls.engine.trainer import Trainer
from xray_cls.models.classifier import XRayClassifier
from xray_cls.utils.io import ensure_dir, load_config
from xray_cls.utils.seed import set_seed


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["seed"])

    device = config["train"]["device"] if torch.cuda.is_available() else "cpu"

    train_dataset = XRayClassificationDataset(
        root_dir=config["data"]["root_dir"],
        split="train",
        transform=get_train_transforms(config["data"]["image_size"]),
    )
    val_dataset = XRayClassificationDataset(
        root_dir=config["data"]["root_dir"],
        split="val",
        transform=get_eval_transforms(config["data"]["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = XRayClassifier(
        model_name=config["model"]["name"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    trainer = Trainer(model, criterion, optimizer, device)

    checkpoint_dir = config["output"]["checkpoint_dir"]
    ensure_dir(checkpoint_dir)

    best_f1 = -1.0

    for epoch in range(config["train"]["epochs"]):
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        print(f"Epoch [{epoch+1}/{config['train']['epochs']}]")
        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "best_f1": best_f1,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)