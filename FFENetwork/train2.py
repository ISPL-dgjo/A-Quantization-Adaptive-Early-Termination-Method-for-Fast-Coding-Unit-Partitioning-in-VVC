import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_tmp import YUVEdgeDataset
from model import Autoenc

def plot_losses(train_losses, val_losses, lrs, save_path='./loss_plot.png'):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(lrs, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = YUVEdgeDataset('./datasets/train/input', './datasets/train/label')
    val_dataset = YUVEdgeDataset('./datasets/val/input', './datasets/val/label')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = Autoenc(in_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    EPOCHS = 250
    train_losses = []
    val_losses = []
    lrs = []

    os.makedirs('./epochs', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels, qps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            qps = qps.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, frame_qp=qps[0].item())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, qps in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, frame_qp=qps[0].item())
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Val Loss: {avg_val_loss:.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        print(f"Current Learning Rate: {current_lr:.6f}")

        torch.save(model.state_dict(), f'./epochs/checkpoint_epoch_{epoch+1}.pth')

        plot_losses(train_losses, val_losses, lrs, save_path='./epochs/loss_lr_plot.png')


if __name__ == '__main__':
    train()
