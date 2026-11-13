"""Compare NIXL and TCP loopback for a GPU tensor shared between two Ray actors."""

import argparse
import json
import socket
import statistics
import time

import ray
import torch
from ray.experimental import set_target_for_ref
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=0.5, enable_tensor_transport=True)
class TransferWorker:
    def __init__(self, numel: int, timeout: float):
        self.tensor = torch.empty(numel, dtype=torch.float32, device="cuda")
        self.host = torch.empty(numel, dtype=torch.float32, pin_memory=True)
        self.buffer = memoryview(self.host.numpy()).cast("B")
        self.timeout = timeout

    def identity(self) -> dict:
        context = ray.get_runtime_context()
        return {"node": context.get_node_id(), "gpu": context.get_accelerator_ids()["GPU"]}

    def listen(self) -> int:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.settimeout(self.timeout)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        return self.listener.getsockname()[1]

    def connect(self, port: int) -> None:
        self.connection = socket.create_connection(("127.0.0.1", port), self.timeout)

    def accept(self) -> None:
        self.connection, _ = self.listener.accept()
        self.connection.settimeout(self.timeout)
        self.listener.close()

    def prepare(self, value: float) -> None:
        self.tensor.fill_(value)
        torch.cuda.synchronize()

    def publish(self) -> dict:
        return {"ref": ray.put(self.tensor, _tensor_transport="nixl")}

    def receive_nixl(self, envelope: dict) -> None:
        set_target_for_ref(envelope["ref"], [self.tensor])
        received = ray.get(envelope["ref"], timeout=self.timeout)
        torch.cuda.synchronize()
        assert received.data_ptr() == self.tensor.data_ptr()

    def send_tcp(self) -> None:
        # CPU staging belongs to the measured GPU-to-GPU transfer.
        self.host.copy_(self.tensor)
        self.connection.sendall(self.buffer)

    def receive_tcp(self) -> None:
        offset = 0
        while offset < len(self.buffer):
            received = self.connection.recv_into(self.buffer[offset:])
            assert received > 0, "TCP connection closed before the tensor was received"
            offset += received
        self.tensor.copy_(self.host)
        torch.cuda.synchronize()

    def verify(self, value: float) -> None:
        # 全量校验放在计时外；每轮值不同，可以发现未执行传输的旧数据。
        assert self.tensor.is_cuda and self.tensor.dtype == torch.float32
        assert torch.all(self.tensor == value).item()


def run_benchmark(args: argparse.Namespace) -> dict:
    # 验证同卡两个进程真实搬运整个 Tensor，并比较相同 GPU→GPU 边界的耗时。
    size_bytes = args.size_mib * 1024**2
    group = placement_group([{"CPU": 2, "GPU": 1}])
    actors = []
    try:
        ray.get(group.ready(), timeout=args.timeout)
        scheduling = PlacementGroupSchedulingStrategy(
            placement_group=group, placement_group_bundle_index=0
        )
        for _ in range(2):
            actors.append(TransferWorker.options(scheduling_strategy=scheduling).remote(
                size_bytes // 4, args.timeout
            ))
        producer, consumer = actors
        locations = ray.get([actor.identity.remote() for actor in actors], timeout=args.timeout)
        assert locations[0] == locations[1], "Both actors must share the same GPU"
        port = ray.get(producer.listen.remote(), timeout=args.timeout)
        ray.get(consumer.connect.remote(port), timeout=args.timeout)
        ray.get(producer.accept.remote(), timeout=args.timeout)

        # 连接和缓冲区初始化不计时，预热单独报告；交替顺序减少先后次序偏差。
        samples = {"nixl": [], "tcp": []}
        for iteration in range(args.warmup + args.repeats):
            modes = ["nixl", "tcp"] if iteration % 2 == 0 else ["tcp", "nixl"]
            for index, mode in enumerate(modes):
                value = float(iteration * 2 + index + 1)
                ray.get(producer.prepare.remote(value), timeout=args.timeout)
                ray.get(consumer.prepare.remote(-1), timeout=args.timeout)
                started = time.perf_counter()
                if mode == "nixl":
                    envelope = ray.get(producer.publish.remote(), timeout=args.timeout)
                    ray.get(consumer.receive_nixl.remote(envelope), timeout=args.timeout)
                else:
                    send = producer.send_tcp.remote()
                    receive = consumer.receive_tcp.remote()
                    ray.get([send, receive], timeout=args.timeout)
                seconds = time.perf_counter() - started

                # 接收完成后检查每个元素，避免将缓存命中、少传或失败计入性能结果。
                ray.get(consumer.verify.remote(value), timeout=args.timeout)
                if mode == "nixl":
                    del envelope
                warmup = iteration < args.warmup
                if not warmup:
                    samples[mode].append(seconds)
                print(json.dumps({"mode": mode, "warmup": warmup, "iteration": iteration,
                                  "seconds": seconds, "verified": True}), flush=True)

        # 有效吞吐包含控制调用和必要拷贝，不解释为网卡带宽或 RDMA 测量。
        summary = {}
        for mode, values in samples.items():
            median = statistics.median(values)
            summary[mode] = {"seconds": values, "median_seconds": median,
                             "effective_gib_per_second": size_bytes / 1024**3 / median}
        return {"bytes": size_bytes, "topology": "same GPU, TCP loopback",
                "locations": locations, "warmup": args.warmup, "repeats": args.repeats,
                "results": summary,
                "tcp_over_nixl": summary["tcp"]["median_seconds"] / summary["nixl"]["median_seconds"]}
    finally:
        for actor in actors:
            ray.kill(actor)
        remove_placement_group(group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address")
    parser.add_argument("--size-mib", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    assert args.size_mib > 0 and args.warmup >= 1 and args.repeats > 0
    assert 0 < args.timeout < float("inf")
    ray.init(address=args.address)
    try:
        print(json.dumps(run_benchmark(args), indent=2), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
