# A-Quantization-Adaptive-Early-Termination-Method-for-Fast-Coding-Unit-Partitioning-in-VVC

# Supplementary code: HTN-based CU partitioning (FFE & PMP)

This repository contains training code and released **checkpoint weights** for the **two neural networks** used in the **HTN-based CU partitioning strategy** from the paper’s **proposed method**.

| Component | Description |
|-----------|-------------|
| **FFE** (`FFENetwork/`) | CNN-based feature extraction network — trains and outputs **frame-level feature maps** |
| **PMP** (`PMPNetwork/`) | Network that takes frame feature maps from FFE and predicts **CU partitioning** |

---

## Repository layout

| Path | Role |
|------|------|
| `FFENetwork/` | Train the **CNN-based** model on YUV input (`train2.py`); extract **frame-level feature maps** with trained weights (`extract_features.py`) |
| `PMPNetwork/` | Train the `SimpleMLP` on frame feature maps (`.pt`) from FFE and per-CU label CSVs (`t_32_32.py`, etc.); export TorchScript (`trace_mlp.py`) |

Paths are relative to the repository root (e.g. `../datasets/train/...`). Adjust them for your environment.

---

## Requirements

- Python 3.9  
- PyTorch (a build compatible with your GPU driver if you use CUDA)  
- Also: `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`  
  - `FFENetwork/dataset_tmp.py` imports `opencv-python` (`cv2`).

---

## 1. FFENetwork

### Training

- Script: `FFENetwork/train2.py`  
- Data: `./datasets/train/input`, `./datasets/train/label`, and the same layout under `val/`  
- Resolution, QP, etc. are parsed from YUV filenames (see regexes and loaders in `dataset_tmp.py`).  
- Model: `Autoenc(in_channels=1)` — trained on a **single Y channel**.  
- Loss: MSE; optimizer: Adam; scheduler: `StepLR`.  
- Checkpoints: `./epochs/checkpoint_epoch_<epoch>.pth`

### Feature extraction (FFE weights)

- Script: `FFENetwork/extract_features.py`  
- Loads weights with **`in_channels=1` (Y only), same as training**, and saves a **full-resolution frame feature map** `[C, H, W]` per input frame as `.pt`.  
- Checkpoint path and output directory are hard-coded at the top of the script; update paths and epoch index when you release artifacts.

---

## 2. PMPNetwork

### Pipeline overview

1. **FFE** writes a **frame-level feature map** `[C, H, W]` as `.pt` for each frame.  
2. Prepare a **CSV** with per-CU partition information dumped from the encoder (or similar), including position `Pos(x,y)`, block size `Block_size(w*h)`, ground-truth partition `Split_mode`, etc.  
3. `FeatureMapCUDataset` crops a **patch per CU** from the frame feature map using CSV coordinates and block sizes, and pairs each patch with the corresponding ground-truth partition label.  
4. For each patch, a vector is built (e.g. via `spatial_pyramid_pool`), combined with normalized block size `(W, H)` from `transform_WH`, and fed to the MLP; training compares **predicted vs. ground-truth partitions**.

### What must stay consistent when changing block size (important)

If you change the CU block size (e.g. 32×32 vs. 64×64), update **all** of the following together:

1. **Data loader (`d_32_32.py`, etc.)**  
   - Adjust filters so **only the intended block sizes** remain, consistent with `Block_size(w*h)` and coordinates in the CSV.  
   - The sample `d_32_32.py` excludes `128×128` only; the filename may not match a filter that selects a specific size (e.g. 32×32). **Verify and edit conditions** to match the block size used in your experiments.

2. **Training script and save directory**  
   - Example: `save_dir = './epochs_32_32'` in `t_32_32.py` — use separate names per block configuration.  
   - Update `block_w`, `block_h`, `epoch_num`, and `model_path` in `trace_mlp.py` for the same experiment.

3. **Number of classes `num_classes`**  
   - `num_classes` in `e2e_model.SimpleMLP` must match the loss and label definition. Keep the `Split_mode` → class-id mapping in the dataset and `NUM_CLASSES` in the training script **aligned** under the same experimental setting.

### TorchScript for inference

- `trace_mlp.py` exports the trained `SimpleMLP` with `torch.jit.trace`.  
- Save path and block-size variables are set inside the script.

---

## Checkpoint files

| Network | Example path (as in code) | Notes |
|---------|---------------------------|--------|
| FFE | `FFENetwork/epochs/checkpoint_epoch_*.pth` | Keep names consistent with `train2.py` / `extract_features.py` |
| PMP | `PMPNetwork/epochs_*_* /epoch_*.pth` | One folder per block-size configuration |

---

## Paper information

- **Title**: A Quantization Adaptive Early Termination Method for Fast Coding Unit Partitioning in VVC   
- **DOI**:  
