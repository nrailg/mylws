"""Transfer a tensor between two Ray actors through NIXL."""

import argparse
import json
import math
import time

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Ray cluster address, e.g. auto; omit to start locally")
    parser.add_argument("--size-mib", type=int, default=16, help="Tensor size in MiB (default: 16)")
    parser.add_argument("--timeout", type=float, default=120, help="Wait timeout in seconds")
    args = parser.parse_args()
    assert args.size_mib > 0
    assert math.isfinite(args.timeout) and args.timeout > 0
    return args


def main():
    args = parse_args()
    @ray.remote(num_cpus=1, num_gpus=0.5, enable_tensor_transport=True)
    class Producer:
        def publish(self, numel: int):
            tensor = torch.full((numel,), 3.25, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            payload = {
                "values": tensor,
                "source_node": ray.get_runtime_context().get_node_id(),
                "source_gpu": ray.get_runtime_context().get_accelerator_ids()["GPU"],
            }
            # NIXL handles the tensor; Ray carries the small Python metadata.
            ref = ray.put(payload, _tensor_transport="nixl")
            # Keep the ref nested so the driver receives a reference, not payload.
            return {'something_else': 'just nothing', "ref": ref}

    @ray.remote(num_cpus=1, num_gpus=0.5, enable_tensor_transport=True)
    class Consumer:
        def consume(self, envelope: dict, numel: int, timeout: float):
            started = time.perf_counter()
            # A nested ObjectRef is not automatically resolved as a task argument.
            payload = ray.get(envelope["ref"], timeout=timeout)
            tensor = payload["values"]
            torch.cuda.synchronize()
            fetch_seconds = time.perf_counter() - started

            assert tensor.shape == (numel,) and tensor.dtype == torch.float32
            assert tensor.is_cuda
            # Verify every element, rather than only comparing a checksum.
            assert torch.all(tensor == 3.25).item()
            destination_node = ray.get_runtime_context().get_node_id()
            destination_gpu = ray.get_runtime_context().get_accelerator_ids()["GPU"]
            assert payload["source_node"] == destination_node
            assert payload["source_gpu"] == destination_gpu
            return {
                "verified": True,
                "transport": "nixl",
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "source_node": payload["source_node"],
                "destination_node": destination_node,
                "source_gpu": payload["source_gpu"],
                "destination_gpu": destination_gpu,
                "fetch_seconds": fetch_seconds,
            }

    ray.init(address=args.address)
    # A single bundle reserves one GPU for both fractional-GPU actors.
    group = placement_group([{"CPU": 2, "GPU": 1}])
    actors = []
    try:
        resources = ray.cluster_resources()
        assert resources.get("CPU", 0) >= 2 and resources.get("GPU", 0) >= 1
        ray.get(group.ready(), timeout=args.timeout)
        scheduling = PlacementGroupSchedulingStrategy(
            placement_group=group, placement_group_bundle_index=0
        )
        producer = Producer.options(scheduling_strategy=scheduling).remote()
        actors.append(producer)
        consumer = Consumer.options(scheduling_strategy=scheduling).remote()
        actors.append(consumer)
        numel = args.size_mib * 1024 * 1024 // 4

        # This ray.get retrieves only {"ref": ObjectRef}; no tensor enters the driver.
        envelope = ray.get(producer.publish.remote(numel), timeout=args.timeout)
        result = ray.get(
            consumer.consume.remote(envelope, numel, args.timeout),
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2))
        # The producer stays alive, and envelope holds the ref, until consumption finishes.
    finally:
        for actor in actors:
            ray.kill(actor)
        remove_placement_group(group)
        ray.shutdown()


if __name__ == "__main__":
    main()
