import torch
import psutil
import gc

def log_memory_stats(key, filename):
    result = {}
    key = key + f"_{filename}_"
    torch.cuda.synchronize()
    if torch.distributed.get_rank() == 0:
        result[key + "_cpu_memory_percent_free_GB"] = psutil.virtual_memory()[4] / 2 ** 30

        result[key + "_memory_allocated_GB"] = torch.cuda.memory_allocated(0) / 2 ** 30
        result[key + "_max_memory_allocated_GB"] = torch.cuda.max_memory_allocated(0) / 2 ** 30

        result[key + "_memory_reserved_GB"] = torch.cuda.memory_reserved(0) / 2 ** 30
        result[key + "_max_memory_reserved_GB"] = torch.cuda.max_memory_reserved(0) / 2 ** 30

        print(f"{result}")

def log_tensors_in_memory(label):
    gc.collect()
    if torch.distributed.get_rank() == 0:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                    print(label, type(obj), type(obj).__name__, obj.size(), obj.device, obj.storage().size())
            except:
                pass