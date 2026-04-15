from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset


class XRayClassificationDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []

        class_to_label = {
            "NORMAL": 0,
            "PNEUMONIA": 1,
        }

        for class_name, label in class_to_label.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, float(label), img_path