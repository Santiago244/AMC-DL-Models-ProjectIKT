import torch
print("cuda_available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    print("alloc (bytes):", torch.cuda.memory_allocated())
    print("reserved (bytes):", torch.cuda.memory_reserved())