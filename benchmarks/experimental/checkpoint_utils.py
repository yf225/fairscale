import hashlib
import json
import logging
import os
import shutil
from typing import List, Optional

import torch

logger = logging.getLogger(__file__)

try:
    from iopath.common.file_io import PathManager

    IOPathPathManager = PathManager()
except ImportError:
    IOPathPathManager = None


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if IOPathPathManager:
            return IOPathPathManager.open(
                path=path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline,
            )
        return open(path, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline,)

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.copy(src_path=src_path, dst_path=dst_path, overwrite=overwrite)
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if IOPathPathManager:
            return IOPathPathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if IOPathPathManager:
            return IOPathPathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def islink(path: str) -> Optional[bool]:
        if not PathManager.path_requires_pathmanager(path):
            return os.path.islink(path)
        return None

    @staticmethod
    def ls(path: str) -> List[str]:
        if IOPathPathManager:
            return IOPathPathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if IOPathPathManager:
            return IOPathPathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if IOPathPathManager:
            return IOPathPathManager.rm(path)
        os.remove(path)
        assert not os.path.exists(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if not PathManager.path_requires_pathmanager(path):
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if IOPathPathManager:
            return IOPathPathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(local_path: str, dst_path: str, overwrite: bool = False, **kwargs) -> None:
        if IOPathPathManager:
            return IOPathPathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def path_requires_pathmanager(path: str) -> bool:
        """Do we require PathManager to access given path?"""
        if IOPathPathManager:
            for p in IOPathPathManager._path_handlers.keys():
                if path.startswith(p):
                    return True
        return False

    @staticmethod
    def supports_rename(path: str) -> bool:
        # PathManager doesn't yet support renames
        return not PathManager.path_requires_pathmanager(path)

    @staticmethod
    def rename(src: str, dst: str):
        os.rename(src, dst)

    """
    ioPath async PathManager methods:
    """

    @staticmethod
    def opena(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        callback_after_file_close=None,
    ):
        """
        Return file descriptor with asynchronous write operations.
        """
        global IOPathPathManager
        return IOPathPathManager.opena(
            path=path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            callback_after_file_close=callback_after_file_close,
        )

    @staticmethod
    def async_close() -> bool:
        """
        Wait for files to be written and clean up asynchronous PathManager.
        NOTE: `PathManager.async_close()` must be called at the end of any
        script that uses `PathManager.opena(...)`.
        """
        global IOPathPathManager
        if IOPathPathManager:
            return IOPathPathManager.async_close()
        return False


def torch_load_cpu(path):
    state = torch.load(path, map_location=torch.device("cpu"))
    # If model was trained with fp16, model from loaded state_dict can be moved to fp16
    if isinstance(state, dict) and "cfg" in state:
        if state["cfg"]["common"]["fp16"] or state["cfg"]["common"]["memory_efficient_fp16"]:
            state["model"] = {k: v.half() for k, v in state["model"].items()}
    return state


def save_json(content, path, indent=4):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent)


def load_json(p):
    return json.load(open(p))


def load_jsonl(path):
    with open(path).read() as jsonl_content:
        result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    return result


def load_and_pop_last_optimizer_state(pth):
    st = torch_load_cpu(pth)
    st.pop("last_optimizer_state", None)
    return st


def convert_consolidated_to_sliced_checkpoints(model_state, rank, checkpoint_path):

    # TODO(anj): Remove hardcoded checkpoint path.
    saved_parameters = {}
    for param_path, param in model_state.items():
        file_path = save_slice(checkpoint_path, param_path, param)
        saved_parameters[param_path] = file_path

    checkpoint_list = {
        "type": "sliced",
        "layers": saved_parameters,
    }
    with PathManager.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint_list, f)


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not PathManager.exists(dir_path):
            PathManager.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def save_slice(checkpoint_path: str, param_path: str, param) -> str:
    """
    Save a slice of the model: a parameter and its associated weights
    - create a folder in which the slice will live
    - save the slice in this folder, with a unique name
    - return the created file name
    """
    checkpoint_sub_folder = os.path.splitext(checkpoint_path)[0] + "_layers"
    makedir(checkpoint_sub_folder)
    hash_name = hashlib.sha1(param_path.encode()).hexdigest()
    file_path = os.path.join(checkpoint_sub_folder, f"{hash_name}.torch")
    file_path = abspath(file_path)
    checkpoint_slice = {"type": "sliced", "weight": param.half()}
    with PathManager.open(file_path, "wb") as f:
        torch.save(checkpoint_slice, f)
    return file_path


result = {}


def init_model_weights(model: FSDP, checkpoint: Dict[str, Any]):
    """
    Given a checkpoint of type "layer_list", initialize the weights of the
    model layer by layer, summoning of parameters on the fly to avoid OOM
    """
    assert checkpoint["type"] == "sliced"

    for path, module in _recursive_visit(model):
        print(f"module {type(module)}")
        for param_path, param in module.named_parameters(prefix=path, recurse=False):
            if torch.distributed.get_rank() == 0:
                monitor_memory(torch.distributed.get_rank(), result, f"{param_path}_before_")
            _init_weight_from_slice(param_path, param.data, checkpoint)
            if torch.distributed.get_rank() == 0:
                monitor_memory(torch.distributed.get_rank(), result, f"{param_path}_after_")
        for buffer_path, buffer in module.named_buffers(prefix=path, recurse=False):
            _init_weight_from_slice(buffer_path, buffer.data, checkpoint)

    for k, v in result.items():
        if torch.distributed.get_rank() == 0:
            print(f"Key:{k} Value {v}")


def _init_weight_from_slice(weight_path: str, weight: torch.Tensor, checkpoint: Dict[str, Any]):
    weight_path = _clean_path(weight_path)
    file_name = checkpoint["layers"].get(weight_path, None)
    assert file_name is not None, f"Could not find buffer: {weight_path}"
    with PathManager.open(file_name, "rb") as f:
        layer_checkpoint = torch.load(f)
    assert layer_checkpoint["type"] == "sliced"
    weight.copy_(layer_checkpoint["weight"])


@contextlib.contextmanager
def null_context():
    yield


def _recursive_visit(model: FullyShardedDataParallel):
    """
    Visit a FSDP model, summoning parameters on the fly
    and releasing them as soon as they are not needed
    This replicates the summoning of parameters as done
    through the forward pass of a FSDP model
    """

    def visit(path, module):
        context = null_context()
        if isinstance(module, FullyShardedDataParallel):
            context = _summon_params(module)

        with context:
            yield path, module
            for name, child in module._modules.items():
                next_path = path + "." + name if path else name
                yield from visit(next_path, child)

    yield from visit("", model)


@contextlib.contextmanager
def _summon_params(module):
    with module.summon_full_params(recurse=False):
        yield


def _clean_path(param_path: str):
    fsdp_names = {"_fsdp_wrapped_module", "_fpw_module"}
    return ".".join([split for split in param_path.split(".") if split not in fsdp_names])
