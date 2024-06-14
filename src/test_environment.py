import torch

try:
    device_index = torch.cuda.current_device()
    print("PyTorch is using GPU")
except RuntimeError:
    device_index = -1
    print("PyTorch is using CPU")
