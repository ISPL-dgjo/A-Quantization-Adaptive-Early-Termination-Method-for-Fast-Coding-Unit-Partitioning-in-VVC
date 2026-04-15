import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FeatureMapCUDataset(Dataset):
    def __init__(self, feature_dir, label_dir, patch_size=128):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.patch_size = patch_size

        self.samples = []
        for fname in sorted(os.listdir(feature_dir)):
            if not fname.endswith('.pt'):
                continue
            name = fname.replace(".pt", "")
            csv_path = os.path.join(label_dir, name + ".csv")
            if not os.path.exists(csv_path):
                print(f" No label for {fname}")
                continue
            self.samples.append((os.path.join(feature_dir, fname), csv_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, csv_path = self.samples[idx]
        feat = torch.load(feat_path)  # [C, H, W]
        df = pd.read_csv(csv_path)

        patches = []
        labels = []
        for _, row in df.iterrows():
            x, y = map(int, row["Pos(x,y)"].split(","))
            w, h = map(int, row["Block_size(w*h)"].split("*"))

            if ((w == 128 and h == 128)):
                continue

            if x + w > feat.shape[2] or y + h > feat.shape[1]:
                continue

            patch = feat[:, y:y+h, x:x+w]
            split = row["Split_mode"].upper()
            SPLIT2ID = {
                "NON_SPLIT": 0,
                "QT"      : 1,
                "BT_H"    : 2,
                "BT_V"    : 3,
                "TT_H"    : 4,
                "TT_V"    : 5,
            }
            if split not in SPLIT2ID:
                continue

            patches.append(patch)
            labels.append(torch.tensor(SPLIT2ID[split], dtype=torch.long))

        return patches, labels