import multiprocessing
import os
import random
import socket
import unittest
from typing import Any

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)


def get_open_port() -> int:
    # Try IPv4 first; if it fails, try IPv6.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def multi_process_parallel(world_size: int, target, *args) -> None:
    distributed_init_port = get_open_port()
    processes = []
    for rank in range(world_size):
        p = multiprocessing.Process(
            target=target, args=(*args, world_size, rank, distributed_init_port)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Process exited with code {p.exitcode}")


def eager_allreduce(test_sizes, test_loop, world_size, rank, distributed_init_port):
    """
    Runs the eager all_reduce test in a separate process.
    """
    # Set the device for the given rank.
    device = torch.device(f"xpu:{rank}")
    torch.xpu.set_device(device)

    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=rank,
        backend="xccl",
    )

    # Initialize model parallelism.
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tensor_model_parallel_group().device_group

    for sz in test_sizes:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for _ in range(test_loop):
                inp1 = torch.randint(
                    1, 16, (sz,), dtype=dtype, device=torch.xpu.current_device()
                )
                out1 = tensor_model_parallel_all_reduce(inp1)
                dist.all_reduce(inp1, group=group)
                torch.testing.assert_close(out1, inp1)


class TestCustomAllReduce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(42)
        # Test tensor sizes from 512B up to 32MB.
        cls.test_sizes = [512, 4096, 32768, 262144, 2097152, 16777216, 33554432]
        cls.world_sizes = [2, 4, 6, 8]
        cls.test_loop = 1

    def test_eager_allreduce(self):
        for world_size in self.world_sizes:
            if world_size > torch.xpu.device_count():
                continue
            multi_process_parallel(
                world_size, eager_allreduce, self.test_sizes, self.test_loop
            )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    unittest.main()
