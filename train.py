from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from model import MNISTNet
transform=transforms.ToTensor()

train_dataset=datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset=datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
model=MNISTNet()
loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2
)

epochs=20
best_val_loss=float("inf")
for epoch in range(epochs):
    model.train()
    epoch_loss=0
    for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        logits=model(X_batch)
        loss=loss_fn(
            logits,
            y_batch
        )
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()


    model.eval()
    val_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            logits=model(X_batch)
            loss=loss_fn(
                logits,
                y_batch
            )
            predictions=torch.argmax(
                logits,
                dim=1
            )
            correct+=(
                predictions==y_batch
            ).sum().item()
            total+=y_batch.size(0)
            val_loss+=loss.item()
        accuracy=100*correct/total
        epoch_loss/=len(train_loader)
        val_loss/=len(test_loader)
        scheduler.step(val_loss)
            
        if val_loss< best_val_loss:
            best_val_loss=val_loss
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "val_loss":val_loss
                    },"best_model.pth"
                )
                
        print(
    f"Epoch {epoch} | "
    f"Train Loss={epoch_loss:.4f} | "
    f"Avarage Val Loss={val_loss:.4f} | "
    f"Avarage  Epoch Loss={epoch_loss:.4f} | "
    f"Accuracy={accuracy:.2f}%"
)