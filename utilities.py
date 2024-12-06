import gc
import time
import torch


def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def check_gpu_memory_usage():
    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
        print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def cleanup(device):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(3)  # some buffer time to allow memory to clear
        torch.cuda.reset_peak_memory_stats()
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def check_network_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")