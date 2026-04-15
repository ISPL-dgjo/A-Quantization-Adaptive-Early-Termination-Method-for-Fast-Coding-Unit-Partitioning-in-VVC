import os
import re
import torch
import numpy as np
from model import Autoenc, QP_halfmask          # ✔ model2 → model
# 학습 때 쓰던 모듈 이름이 dataset_tmp.py라면 경로에 맞춰 주세요
from dataset_tmp import YUVEdgeDataset          # ✔ 경로 확인

# ---------- 유틸 ----------
def extract_resolution_and_qp(filename: str):
    """example:  PeopleOnStreet_256x128_15_22.yuv → (256,128,22)"""
    m = re.search(r'_(\d+)x(\d+)_\d+_(\d{2})\.yuv$', filename)
    if not m:
        raise ValueError(f"Cannot parse {filename}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def load_y_frame_420_8bit(path: str, w: int, h: int) -> np.ndarray:
    """간단히 Y(8-bit)만 읽어 [H,W] float32(0-1) 반환"""
    with open(path, "rb") as f:
        y = np.frombuffer(f.read(w * h), dtype=np.uint8)\
              .reshape(h, w)\
              .astype(np.float32) / 255.0
    return y

def load_y_frame_400_10bit_le(yuv_path, width, height):
        """
        YUV 4:0:0 10bit LE 파일에서 단일 프레임의 Y 성분만 읽어 [1, H, W] 형태의 PyTorch 텐서로 반환.
        U/V 성분은 존재하지 않음.
        """
        y_samples = width * height
        y_bytes_needed = y_samples * 2  # 2바이트 per sample

        with open(yuv_path, 'rb') as f:
            y_bytes = f.read(y_bytes_needed)

        # 16비트로 읽고 10비트 데이터로 해석
        y = np.frombuffer(y_bytes, dtype=np.uint16).reshape((height, width))

        # 10비트 정규화: [0, 1023] → [0.0, 1.0]
        y = y.astype(np.float32) / 1023.0

        # PyTorch 텐서로 변환, [1, H, W] (channel-first)
        # y_tensor = torch.from_numpy(y).unsqueeze(0)

        return y
# -------------------------

def extract_feature_maps():
    in_dir  = "../mlp_2/datasets/test/input"
    out_dir = "../mlp_2/datasets/test/features"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습(train2.py)과 동일하게 단일 Y 채널(in_channels=1)로 초기화
    model = Autoenc(in_channels=1).to(device)
    model.load_state_dict(
        torch.load("./epochs/checkpoint_epoch_276.pth", map_location=device)
    )
    model.eval()

    for fname in sorted(f for f in os.listdir(in_dir) if f.endswith(".yuv")):
        print(f"🔍 {fname}")
        w, h, qp = extract_resolution_and_qp(fname)
        y = load_y_frame_420_8bit(os.path.join(in_dir, fname), w, h)

        y_tensor    = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        

        with torch.no_grad():
            # --------- Autoenc forward (마지막 cbam까지만 복제) ---------
            grad_x = torch.nn.functional.conv2d(y_tensor, model.weight_x, padding=1)
            grad_y = torch.nn.functional.conv2d(y_tensor, model.weight_y, padding=1)

            input_x = y_tensor
            branch1 = model.relu(model.conv1(input_x))
            out1 = model.ResB3(model.ResB2(model.ResB1(branch1)))
            out1 = model.relu(out1)

            edge   = torch.cat([grad_x, grad_y], dim=1)
            branch2 = model.relu(model.conv2(edge))
            out2 = model.ResB22(model.ResB11(branch2))
            out2 = model.relu(out2)

            out = model.conv3(torch.cat([out1, out2], dim=1))
            out = QP_halfmask(out, qp)
            feature = model.cbam(out).squeeze(0).cpu()                          # [32,H,W]
            # -------------------------------------------------------------

        save_path = os.path.join(out_dir, fname.replace(".yuv", ".pt"))
        torch.save(feature, save_path)
        print(f"saved → {save_path}")

if __name__ == "__main__":
    extract_feature_maps()
