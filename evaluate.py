from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
from model import MNISTNet

transform=transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
model = MNISTNet()

checkpoint = torch.load(
    "best_model.pth"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

all_preds=[]
all_labels=[]

correct=0
total=0
with torch.no_grad():
    for X_batch,y_batch in test_loader:
        logits=model(X_batch)

        predictions=torch.argmax(
            logits,
            dim=1
        )
        correct+=(
            predictions==y_batch
        ).sum().item()
        total+=y_batch.size(0)

        all_preds.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            y_batch.cpu().numpy()
        )

accuracy = 100 * correct / total

print(
    f"Test Accuracy: {accuracy:.2f}%"
)

print(
    classification_report(
        all_labels,
        all_preds
    )
)
ConfusionMatrixDisplay.from_predictions(
    all_labels,
    all_preds
)
plt.savefig(
    "confusion_matrix.png",
    bbox_inches="tight"
)

plt.show()