import gc
import time
import torch
import json
import random

class IterReadDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path:str, total_lines: int, world_size:int, rank:int, offset_seed:int = 0, buffer_size:int = 0):
        super().__init__()
        self.file_path = file_path
        self.world_size = world_size
        self.rank = rank
        self.buffer_size = buffer_size

        # exchange rank index by offset_seed, one of shuffle approaches
        if world_size > 1 and offset_seed != 0 and offset_seed < world_size:
            rank = (rank + offset_seed) % world_size


        # to calculate initial position for all sub_process
        self.per_worker = total_lines // world_size
        self.iter_start = rank * self.per_worker

        if rank == self.world_size - 1:
            # let last rank deal with the remaining data
            self.iter_end = total_lines
        else:
            self.iter_end = self.iter_start + self.per_worker

    def __len__(self):
        return self.iter_end - self.iter_start

    def __iter__(self):
        buffer = []  # Buffer for shuffle

        with open(self.file_path, "r", encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i < self.iter_start:
                    continue
                if i >= self.iter_end:
                    break

                data = json.loads(line)

                if self.buffer_size > 0:
                    # Add to buffer
                    buffer.append(data)
                    if len(buffer) >= self.buffer_size:
                        # Shuffle and yield
                        random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []
                else:
                    # Yield data directly without shuffle
                    yield data

            # Yield remaining data in buffer
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    yield item




def memory_tracking():
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