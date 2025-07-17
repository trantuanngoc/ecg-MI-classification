import torch
import random

def random_shift_1d(tensor, max_shift_percent=0.1, random_seed=30):
    batch_size, length = tensor.shape
    max_shift = int(length * max_shift_percent)
    random.seed(random_seed)
    shift = random.randint(-max_shift, max_shift)

    if shift == 0:
        return tensor.clone()

    # Create zero-padded tensor of the same shape
    shifted = torch.zeros_like(tensor)

    if shift > 0:
        shifted[..., shift:] = tensor[..., :-shift]
    else:  # shift < 0
        shifted[..., :shift] = tensor[..., -shift:]

    return shifted

def roll_1d(tensor, max_shift_percent=0.1):
    length = tensor.shape[0]
    max_shift = int(length * max_shift_percent)
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()

    return torch.roll(tensor, shifts=shift, dims=0)

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal