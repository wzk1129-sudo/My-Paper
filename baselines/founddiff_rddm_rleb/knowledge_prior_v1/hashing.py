"""Exact dtype-safe hashing for mixed-precision model state tensors."""
import hashlib
import torch

def exact_state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        h.update(raw.numpy().tobytes())
    return h.hexdigest()
