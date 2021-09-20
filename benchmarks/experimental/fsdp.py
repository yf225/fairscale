# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict
from functools import reduce
import functools
import gc
import logging
import math
import operator
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import torch.multiprocessing as mp
import psutil

import gc
import operator as op

from benchmarks.datasets.wikitext2_data import get_real_dataloaders as get_real_wikitext2_dataloaders
from benchmarks.datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from benchmarks.golden_configs.lm_wikitext2 import Pipe as lm_wikitext2
from benchmarks.models import transformer_lm
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


RPC_PORT = 29501


def init_nvml_profiling(rank):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(rank)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_model_and_optimizer(args, device, benchmark_config, model_config):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, model_config)

    lr = benchmark_config["lr"]

    def make_adam(params):
        return Adam(params, lr=lr)

    optimizer = make_adam
    return model, optimizer


def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    ndecoder = config["num_decoder_layers"]

    if args.lazy_construction:
        layers = [
            LazyModule(lambda: transformer_lm.EmbeddingLayer(vocab_size, ninp, initrange)),
            LazyModule(lambda: transformer_lm.PositionalEncodingLayer(ninp, dropout)),
        ]
        for _ in range(ndecoder):
            layers.append(LazyModule(lambda: transformer_lm.TransformerDecoderLayer(ninp, nhead, nhid, dropout)))

        layers.append(LazyModule(lambda: transformer_lm.LinearLayer(ninp, vocab_size, initrange)))
        model = layers
    else:
        model = transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)

    return model


def get_tensors_by_size_bucket():

    size_buckets = defaultdict(int)
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    return size_buckets


def log_number_of_parameters(model):

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if hasattr(model, "group"):
        total = torch.Tensor([num_params])
        if torch.cuda.is_available():
            total = total.cuda()
        torch.distributed.all_reduce(total, group=model.group)
        print(
            f"training model, #params = {num_params/10**6}M, group: {model.group.rank()}, grank:"
            f" {torch.distributed.get_rank()}, sizes {model.group.size()}"
        )
        torch.distributed.barrier()
        if model.group.rank() == 0:
            print(f"total #prams = {total.item()}")
    else:
        print(f"training model, #params = {num_params/10**6}M")


def get_device(model, index):
    if isinstance(model, DDP):
        model = model.module

    if not torch.cuda.is_available():
        return torch.device("cpu")
    if hasattr(model, "devices"):
        return model.devices[index]
    else:
        return torch.cuda.current_device()


def get_fake_dataloader(lm_dataloader_len, args):
    fake_input = {"input": torch.zeros(args.batch_size)}

    class FakeDataset:
        def __getitem__(self, index):
            return fake_input

        def __len__(self):
            return lm_dataloader_len

    return FakeDataset()


def train(model_config, model, benchmark_config, model_specs, args):
    lm_dataloader, _, _ = model_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = model_specs["vocab_size"]
    optimizer = model_config["optimizer"]

    model.train()
    log_number_of_parameters(model)

    total_loss = 0.0
    word_counter = 0

    optimizer = optimizer(model.parameters())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    total_tokens = 0
    total_tokens_per_log_interval = 0
    bptt = 2
    start_time = time.time()
    epoch_start_time = 0.0

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target

    for i, batch in enumerate(lm_dataloader):
        if i == 1:
            epoch_start_time = time.time()

        source, target = get_batch(batch)
        if args.max_batch and i > args.max_batch:
            break

        if i > 0:
            total_tokens += source.numel()

        optimizer.zero_grad()
        input = source.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), model_specs["clip_value"])
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        total_tokens_per_log_interval += source.numel()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if dist.get_rank() == dist.get_world_size() - 1:
                logging.debug(
                    "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                        i, total_tokens_per_log_interval / elapsed, cur_loss, math.exp(cur_loss)
                    )
                )
            total_tokens_per_log_interval = 0
            total_loss = 0
            start_time = time.time()

    if epoch_start_time != 0:
        torch.cuda.synchronize()
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    if dist.get_rank() == dist.get_world_size() - 1:
        return wps, loss.item()
    else:
        return 0.0, 0.0

def eval(model_config, model, benchmark_config, model_specs, args):
    print(f"Benchmarking Eval..")
    info = init_nvml_profiling(torch.distributed.get_rank())
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"1---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
    lm_dataloader, _, _ = model_config["data"]
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"2 ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
    criterion = benchmark_config["criterion"]
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"3 ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
    vocab_size = model_specs["vocab_size"]
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"4 ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
        print("----------------------------------------------------------")
    model.eval()
    # log_number_of_parameters(model)

    total_loss = 0.0
    
    total_tokens = 0
    total_tokens_per_log_interval = 0
    start_time = time.time()
    epoch_start_time = 0.0

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target

    info = init_nvml_profiling(torch.distributed.get_rank())
    for i, batch in enumerate(lm_dataloader):
        if torch.distributed.get_rank() == 0:
            print("----------------------------------------------------------")
            print(f"5 ---- CPU: {psutil.virtual_memory()}")
            print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
            print(f"Total memory: {info.total/2**30}")
            print(f"Free memory: {info.free/2**30}")
            print(f"Used memory: {info.used/2**30}")
            print("----------------------------------------------------------")
        if i == 1:
            epoch_start_time = time.time()

        source, target = get_batch(batch)
        if args.max_batch and i > args.max_batch:
            break

        if i > 0:
            total_tokens += source.numel()
 

        if torch.distributed.get_rank() == 0:
            print("----------------------------------------------------------")
            print(f"Before Eval inside loop {i} ---- CPU: {psutil.virtual_memory()}")
            print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
            print(f"Total memory: {info.total/2**30}")
            print(f"Free memory: {info.free/2**30}")
            print(f"Used memory: {info.used/2**30}")
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size(), obj.device)
                except:
                    pass
            print("----------------------------------------------------------")
            print(f"\n\n EVAL begins for iteration {i}")
        
        input = source.cuda()
        target = target.cuda()

        with torch.no_grad():
            output = model(input)

        if torch.distributed.get_rank() == 0:
            
            print("----------------------------------------------------------")
            print(f"After Eval inside loop {i} ---- CPU: {psutil.virtual_memory()}")
            print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
            print(f"Total memory: {info.total/2**30}")
            print(f"Free memory: {info.free/2**30}")
            print(f"Used memory: {info.used/2**30}")
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size(), obj.device)
                except:
                    pass
            print("----------------------------------------------------------")
            
            print(f"\n\n EVAL ends for iteration {i}")

        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        total_loss += loss.item()
        log_interval = 1
        total_tokens_per_log_interval += source.numel()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if dist.get_rank() == 0:
                print(
                    "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                        i, total_tokens_per_log_interval / elapsed, cur_loss, math.exp(cur_loss)
                    )
                )
            total_tokens_per_log_interval = 0
            total_loss = 0
            start_time = time.time()
        

    if epoch_start_time != 0:
        torch.cuda.synchronize()
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    if dist.get_rank() == dist.get_world_size() - 1:
        return wps, loss.item()
    else:
        return 0.0, 0.0



def get_number_of_words(data):
    return data.size()[0] * data.size()[1]


def benchmark_language_model(model_config, model, benchmark_config, model_specs, args):
    golden_config = get_golden_config(args.model_name, args)
    epoch = benchmark_config["epochs"]
    start_time = time.time()
    if dist.get_rank() == dist.get_world_size() - 1:
        logging.debug("-" * 110)
        logging.debug("| start of epoch {:1d}".format(epoch))
        logging.debug("-" * 110)
    if args.train:
        wps, loss = train(model_config, model, benchmark_config, model_specs, args)
    else:
        wps, loss = eval(model_config, model, benchmark_config, model_specs, args)
    elapsed_time = time.time() - start_time
    if dist.get_rank() == dist.get_world_size() - 1:
        logging.debug("-" * 110)
        logging.debug("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
        logging.debug("-" * 110)
        logging.debug("Throughput(wps) is {:.2f}.".format(wps))
    logging.debug(
        "Peak allocated bytes on cuda:{}: {:4f}GB".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2 ** 30
        )
    )


def get_synthetic_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloader for synthetic data."""

    if args.model_name == "lm":
        return get_synthetic_wikitext2_dataloaders(args, benchmark_config, model_specs)
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_real_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloaders for real data."""

    if args.model_name == "lm":
        data = get_real_wikitext2_dataloaders(args, benchmark_config, model_specs)
        ntokens, train_dataloader, valid_dataloader, test_dataloader = data
        model_specs["vocab_size"] = ntokens
        return train_dataloader, valid_dataloader, test_dataloader
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def create_model_config(args, benchmark_config=None, model_specs=None):
    """Return a dict with the given model, dataset and optimizer."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.use_synthetic_data:
        dataloader_fn = get_synthetic_dataloaders
    else:
        dataloader_fn = get_real_dataloaders

    data = dataloader_fn(args, device, benchmark_config, model_specs)
    model, optimizer = get_model_and_optimizer(args, device, benchmark_config, model_specs)
    return {
        "model": model,
        "optimizer": optimizer,
        "data": data,
    }


def create_benchmark_config(model_name):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_benchmark_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_model_specs(model_name):
    """Return a dict with configurations required for configuring `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_model_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_golden_config(model_name, args):
    """Return a dict with the golden data for throughput and memory usage."""

    if model_name == "lm":
        return lm_wikitext2.get_golden_real_stats()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_single_process(rank, args):
    """Benchmark a given model using a single process and multiple devices."""
    torch.cuda.set_device(rank)
    info = init_nvml_profiling(rank)
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=torch.cuda.device_count(), init_method=init_method_pgroup)

    torch.distributed.all_reduce(torch.ones((1, 1)).cuda())

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0
    init_random_seed(0)
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"Before Model Loading ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass
        print("----------------------------------------------------------")
    benchmark_config = create_benchmark_config(args.model_name)
    model_specs = get_model_specs(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
    model = model_config["model"]

    model = model.cpu()

    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"After Model Loading Before FSDP Wrap ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass
        print("----------------------------------------------------------")

    my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=args.min_wrap_params)
    config = {}
    config["flatten_parameters"] = True
    config["ssd_offload"] = args.ssd_offload
    if args.auto_wrap:
        with enable_wrap(wrapper_cls=FSDP, **config):
            fsdp_model = auto_wrap(model, auto_wrap_policy=my_auto_wrap_policy)
    else:
        fsdp_model = FSDP(model, ssd_offload=config["ssd_offload"])
    if torch.distributed.get_rank() == 0:
        print("----------------------------------------------------------")
        print(f"After FSDP Wrap ---- CPU: {psutil.virtual_memory()}")
        print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
        print(f"Total memory: {info.total/2**30}")
        print(f"Free memory: {info.free/2**30}")
        print(f"Used memory: {info.used/2**30}")
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass
        print("----------------------------------------------------------")
    if args.dry_run:
        if args.train:
            train(model_config, fsdp_model, benchmark_config, model_specs, args)
        else:
            if torch.distributed.get_rank() == 0:
                print("----------------------------------------------------------")
                print(f"0 ---- CPU: {psutil.virtual_memory()}")
                print(f"GPU: torch.cuda.max_memory_allocated:{torch.cuda.max_memory_allocated(0)/2**30} torch.cuda.memory_allocated:{torch.cuda.memory_allocated(0)/2**30}")
                print(f"Total memory: {info.total/2**30}")
                print(f"Free memory: {info.free/2**30}")
                print(f"Used memory: {info.used/2**30}")
            eval(model_config, fsdp_model, benchmark_config, model_specs, args)
    else:
        benchmark_language_model(model_config, fsdp_model, benchmark_config, model_specs, args)
    
    torch.cuda.empty_cache()
    if torch.distributed.get_rank() == 0:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(rank)
        info = nvmlDeviceGetMemoryInfo(handle)
        print("\n\n Total memory:", info.total/2**30)
        print("Free memory:", info.free/2**30)
        print("Used memory:", info.used/2**30)
        print(f"After eval CPU: {psutil.virtual_memory()} GPU {torch.cuda.max_memory_allocated(rank)/2**30}")
        torch.cuda.empty_cache()
        gc.collect()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size(), obj.device)
            except:
                pass

parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument(
    "--lazy_construction", action="store_true", default=False, help="Number of decoder layers in the model"
)
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Test training instead of eval. This is a stopgap benchmark till we support SSD offload + training.",
)
parser.add_argument("--min_wrap_params", type=int, default=1e8, help="Maximum number of params before we FSDP wrap a module.")
parser.add_argument("--max_batch", type=int, default=4, help="Max number of batches")
parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
parser.add_argument(
    # TODO(anj-s): In the process of adding more models and hence the requirement for a flag.
    "--model_name",
    default="lm",
    help="Language Model(LM) used to benchmark FSDP.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Display additional debug information")
parser.add_argument("--auto_wrap", action="store_true", default=False, help="Use auto_wrap with FSDP")
parser.add_argument("--ssd_offload", action="store_true", default=False, help="Use ssd_offload FSDP")

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    logging.info(f"Running single process benchmark with args: {args}")

    if torch.cuda.device_count() == 0:
        raise RuntimeError("This benchmark requires GPUs and does not support CPU only training.")

    mp.spawn(
        benchmark_single_process,  # type: ignore
        args=(args,),
        nprocs=torch.cuda.device_count(),
        join=True,
    )
