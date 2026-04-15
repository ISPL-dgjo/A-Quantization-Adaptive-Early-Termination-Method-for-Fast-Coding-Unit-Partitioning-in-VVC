# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.fc1 = nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max = F.adaptive_max_pool2d(x, 1)
        out = self.fc2(self.relu(self.fc1(avg))) + self.fc2(self.relu(self.fc1(max)))
        return torch.sigmoid(out)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        return torch.sigmoid(self.compress(x))

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = x * self.ChannelGate(x)
        x = x * self.SpatialGate(x)
        return x

def QP_halfmask(input, QP):
    masked_qp = (QP / 63) + 0.5
    out = input.clone()
    B, C, H, W = out.size()
    half = C // 2
    out[:, :half, :, :] *= masked_qp
    return out

class Autoenc(nn.Module):
    def __init__(self, in_channels=2):
        super(Autoenc, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.relu = nn.ReLU()

        self.ResB1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        self.ResB2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.ResB3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )

        self.ResB11 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        self.ResB22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )

        self.cbam = CBAM(gate_channels=32, reduction_ratio=16)

        self.last = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('weight_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('weight_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x, frame_qp=22):
        grad_x = F.conv2d(x[:, 0:1], self.weight_x, padding=1)
        grad_y = F.conv2d(x[:, 0:1], self.weight_y, padding=1)

        input_x = x
        branch1 = self.relu(self.conv1(input_x))
        out1 = self.ResB1(branch1)
        out1 = self.ResB2(out1)
        out1 = self.ResB3(out1)
        out1 = self.relu(out1)

        edge = torch.cat([grad_x, grad_y], dim=1)
        branch2 = self.relu(self.conv2(edge))
        out2 = self.ResB11(branch2)
        out2 = self.ResB22(out2)
        out2 = self.relu(out2)

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = QP_halfmask(out, frame_qp)
        out = self.cbam(out)

        return self.last(out)
