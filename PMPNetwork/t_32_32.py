import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PMP.e2e_model import spatial_pyramid_pool, SimpleMLP
from d_32_32 import FeatureMapCUDataset
import math
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#Focal Loss 
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape [num_classes]
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

train_feat_dir = '../datasets/train/features'
train_csv_dir  = '../datasets/train/csv'
val_feat_dir   = '../datasets/val/features'
val_csv_dir    = '../datasets/val/csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size    = 32
num_epochs    = 200
learning_rate = 1e-4
max_grad_norm = 1.0

save_dir = './epochs_32_32' # Block size
os.makedirs(save_dir, exist_ok=True)

train_set = FeatureMapCUDataset(train_feat_dir, train_csv_dir)
val_set   = FeatureMapCUDataset(val_feat_dir, val_csv_dir)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)


all_labels = []
for _, labels in train_loader:
    for label_seq in labels:
        for label in label_seq:
            all_labels.append(label.item())

NUM_CLASSES = 2
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=all_labels
)
alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

model = SimpleMLP(input_dim=674, num_classes=NUM_CLASSES).to(device)
criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def transform_WH(W, H):
    def transform(x):
        return (math.log2(x) - 1) / 5
    return torch.tensor([[transform(W), transform(H)]], dtype=torch.float32)

best_val_acc = 0
early_stop_counter = 0
early_stop_patience = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for patches, labels in train_loader:
        for i in range(len(patches)):
            patches_i, labels_i = patches[i], labels[i]
            for patch, label in zip(patches_i, labels_i):
                patch = patch.to(device)
                label = label.to(device)

                C, H, W = patch.size()
                b_size = transform_WH(W, H).to(device)
                spp_feat = spatial_pyramid_pool(patch)
                mlp_input = torch.cat([spp_feat.to(device), b_size], dim=1)

                outputs = model(mlp_input)
                loss = criterion(outputs, label.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_samples += 1
                pred = outputs.argmax(dim=1).item()
                correct += int(pred == label.item())

    train_loss = total_loss / total_samples
    train_acc = correct / total_samples * 100


    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for patches, labels in val_loader:
            for i in range(len(patches)):
                patches_i, labels_i = patches[i], labels[i]
                for patch, label in zip(patches_i, labels_i):
                    patch = patch.to(device)
                    label = label.to(device)

                    C, H, W = patch.size()
                    b_size = transform_WH(W, H).to(device)
                    spp_feat = spatial_pyramid_pool(patch)
                    mlp_input = torch.cat([spp_feat.to(device), b_size], dim=1)

                    outputs = model(mlp_input)
                    val_total += 1
                    val_correct += int(outputs.argmax(dim=1).item() == label.item())

    val_acc = val_correct / val_total * 100
    scheduler.step(train_loss)

    print(f"[Epoch {epoch+1:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || Val Acc: {val_acc:.2f}%")
    print("-" * 60)

    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))