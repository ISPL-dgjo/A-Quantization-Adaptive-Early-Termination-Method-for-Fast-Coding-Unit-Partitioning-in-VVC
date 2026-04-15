import torch
import os
import math
from PMP.e2e_model import SimpleMLP


block_w = 64
block_h = 64
NUM_CLASSES = 2
epoch_num = 53

model_path = "./epochs_" + str(block_w) + "_" +  str(block_h) + "/epoch_" + str(epoch_num) + ".pth" 
save_path  = "./trace_block/model_" + str(block_w) + "_" + str(block_h) +  ".pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def transform_WH(W, H):
    def transform(x):
        if x <= 0:
            raise ValueError("Input must be greater than 0.")
        return (math.log2(x) - 1) / 5
    return torch.tensor([[transform(W), transform(H)]], dtype=torch.float32)

model = SimpleMLP(input_dim=674, num_classes=NUM_CLASSES)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 672)
b_size = transform_WH(block_w, block_h)
dummy_input = torch.cat([dummy_input, b_size], dim=1)

traced_model = torch.jit.trace(model, dummy_input)

traced_model.save(save_path)

print(f"Traced MLP saved to: {save_path}")
