"""Transfer a tensor between two Ray actors through NIXL."""

import argparse
import json
import math
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", help="Ray cluster address, e.g. auto; omit to start locally")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size-mib", type=int, default=16, help="Tensor size in MiB (default: 16)")
    parser.add_argument("--cross-node", action="store_true", help="Require two different Ray nodes")
    parser.add_argument("--timeout", type=float, default=120, help="Wait timeout in seconds")
    args = parser.parse_args()
    if args.size_mib <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--size-mib and --timeout must be positive and finite")
    return args


def main():
    args = parse_args()
    # Delayed imports let --help work without the Linux runtime installed.
    import ray
    import torch
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=1, enable_tensor_transport=True)
    class Producer:
        def publish(self, numel, device):
            tensor = torch.full((numel,), 3.25, dtype=torch.float32, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            payload = {
                "values": tensor,
                "source_node": ray.get_runtime_context().get_node_id(),
            }
            # NIXL handles the tensor; Ray carries the small Python metadata.
            ref = ray.put(payload, _tensor_transport="nixl")
            # Keep the ref nested so the driver receives a reference, not payload.
            return {"ref": ref}

    @ray.remote(num_cpus=1, enable_tensor_transport=True)
    class Consumer:
        def consume(self, envelope, numel, device, timeout):
            started = time.perf_counter()
            # A nested ObjectRef is not automatically resolved as a task argument.
            payload = ray.get(envelope["ref"], timeout=timeout)
            tensor = payload["values"]
            if tensor.is_cuda:
                torch.cuda.synchronize()
            fetch_seconds = time.perf_counter() - started

            if tensor.shape != (numel,) or tensor.dtype != torch.float32:
                raise ValueError(f"Unexpected shape/dtype: {tensor.shape}, {tensor.dtype}")
            if tensor.device.type != device:
                raise ValueError(f"Expected {device}, received {tensor.device}")
            # Verify every element, rather than only comparing a checksum.
            if not torch.all(tensor == 3.25).item():
                raise ValueError("Transferred tensor failed verification")
            return {
                "verified": True,
                "transport": "nixl",
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "source_node": payload["source_node"],
                "destination_node": ray.get_runtime_context().get_node_id(),
                "fetch_seconds": fetch_seconds,
            }

    ray.init(address=args.address)
    actors = []
    try:
        gpu_count = 1 if args.device == "cuda" else 0
        options = [{"num_gpus": gpu_count}, {"num_gpus": gpu_count}]
        if args.cross_node:
            nodes = [
                node for node in ray.nodes()
                if node["Alive"]
                and node["Resources"].get("CPU", 0) >= 1
                and node["Resources"].get("GPU", 0) >= gpu_count
            ]
            if len(nodes) < 2:
                raise RuntimeError("--cross-node requires two live nodes with suitable CPU/GPU resources")
            for actor_options, node in zip(options, nodes[:2]):
                actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
        else:
            resources = ray.cluster_resources()
            if resources.get("CPU", 0) < 2 or resources.get("GPU", 0) < 2 * gpu_count:
                raise RuntimeError("Requires 2 CPUs; CUDA mode also requires 2 GPUs")

        producer = Producer.options(**options[0]).remote()
        actors.append(producer)
        consumer = Consumer.options(**options[1]).remote()
        actors.append(consumer)
        numel = args.size_mib * 1024 * 1024 // 4

        # This ray.get retrieves only {"ref": ObjectRef}; no tensor enters the driver.
        envelope = ray.get(producer.publish.remote(numel, args.device), timeout=args.timeout)
        result = ray.get(
            consumer.consume.remote(envelope, numel, args.device, args.timeout),
            timeout=args.timeout,
        )
        if args.cross_node and result["source_node"] == result["destination_node"]:
            raise RuntimeError("Expected a cross-node transfer")
        print(json.dumps(result, indent=2))
        # The producer stays alive, and envelope holds the ref, until consumption finishes.
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()
