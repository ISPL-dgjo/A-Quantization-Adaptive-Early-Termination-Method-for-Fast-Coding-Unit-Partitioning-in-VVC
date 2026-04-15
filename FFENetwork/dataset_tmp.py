# dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import re
import cv2

class YUVEdgeDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.file_list = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.file_list)

    def parse_resolution(self, filename):
        match = re.search(r'_(\d+)x(\d+)_', filename)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
        raise ValueError(f"Cannot parse resolution from filename: {filename}")

    # def read_yuv420_10bit(self, filepath, width, height):
    #     with open(filepath, 'rb') as f:
    #         y = np.frombuffer(f.read(width * height * 2), dtype=np.uint16)
    #     y = y.reshape((height, width))
    #     y = y.astype(np.float32) / 1023.0
    #     return y
    
    # def load_y_frame_10bit_le(self, yuv_path, width, height):
    #     """
    #     
    #     """
    #     y_samples = width * height
    #     y_bytes_needed = y_samples * 2  # 16bit 

    #     with open(yuv_path, 'rb') as f:
    #         y_bytes = f.read(y_bytes_needed)
        
    #     print(yuv_path, width, height)

    #     y = np.frombuffer(y_bytes, dtype=np.uint16).reshape((height, width))
    #     y_normalized = y.astype(np.float32) / 1023.0
    #     # y_tensor = torch.from_numpy(y_normalized).unsqueeze(0)  # [1, H, W]

    #     return y_normalized
    
    def load_y_frame_420_8bit(self, yuv_path, width, height):

        y_samples = width * height
        frame_size = y_samples * 3 // 2  # YUV420: Y + U/4 + V/4

        with open(yuv_path, 'rb') as f:
            y_bytes = f.read(y_samples)  

        
        y = np.frombuffer(y_bytes, dtype=np.uint8).reshape((height, width))

        
        y = y.astype(np.float32) / 255.0

        

        return y

    def load_y_frame_400_10bit_le(self, yuv_path, width, height):
        y_samples = width * height
        y_bytes_needed = y_samples * 2  

        with open(yuv_path, 'rb') as f:
            y_bytes = f.read(y_bytes_needed)

        y = np.frombuffer(y_bytes, dtype=np.uint16).reshape((height, width))
        y = y.astype(np.float32) / 1023.0


        return y

    def extract_qp(self, filename):
        match = re.search(r'_(\d{2})\.yuv$', filename)
        return int(match.group(1)) if match else 22

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        input_path = os.path.join(self.input_dir, filename)
        label_fname = filename.split(".yuv")[0] + "_bs.yuv"
        label_path = os.path.join(self.label_dir, label_fname)
        # grid_dir = "./datasets/grid"

        width, height = self.parse_resolution(filename)
        # grid_path = os.path.join(grid_dir, grid_name)

        y_img = self.load_y_frame_420_8bit(input_path, width, height)
        label_img = self.load_y_frame_400_10bit_le(label_path, width, height)
        # grid_img = self.load_y_frame_400_10bit_le(grid_path, width, height)

        # Sobel edge
        # sobelx = cv2.Sobel(y_img, cv2.CV_32F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(y_img, cv2.CV_32F, 0, 1, ksize=3)

        y_tensor = torch.from_numpy(y_img).unsqueeze(0)
        # sobelx_tensor = torch.from_numpy(sobelx).unsqueeze(0)
        # sobely_tensor = torch.from_numpy(sobely).unsqueeze(0)
        input_tensor = y_tensor

        # grid_tensor = torch.from_numpy(grid_img).unsqueeze(0)

        label_tensor = torch.from_numpy(label_img).unsqueeze(0)

        qp = self.extract_qp(filename)

        return input_tensor, label_tensor, qp
