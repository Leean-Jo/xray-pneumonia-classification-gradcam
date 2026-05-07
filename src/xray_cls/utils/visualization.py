from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true,
    y_prob,
    save_path: str,
    threshold: float = 0.5,
) -> None:
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"],
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()