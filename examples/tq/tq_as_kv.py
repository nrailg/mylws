import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict


def main() -> None:
    ray.init(address="local")
    try:
        tq.init()
        keys = ["sample_0", "sample_1"]
        partition = "kv_example"
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        # 每个 key 对应 batch 中的一行；返回的 meta 不包含 Tensor 本体。
        meta = tq.kv_batch_put(
            keys=keys,
            partition_id=partition,
            fields=TensorDict({"input_ids": input_ids}, batch_size=[2]),
        )
        print("meta:", meta)
        assert meta.keys == keys

        data = tq.kv_batch_get(
            keys=meta.keys, partition_id=meta.partition_id, select_fields=["input_ids"]
        )
        # TQ 将每个样本的向量打包成 nested tensor，逐样本校验。
        for i in range(len(keys)):
            assert torch.equal(data["input_ids"][i], input_ids[i])
        print("input_ids:", data["input_ids"])

        # 给相同样本补充字段，原来的 input_ids 仍然保留。
        scores = torch.tensor([0.5, 1.0])
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition,
            fields=TensorDict({"score": scores}, batch_size=[2]),
        )
        data = tq.kv_batch_get(keys=keys, partition_id=partition)
        for i in range(len(keys)):
            assert torch.equal(data["input_ids"][i], input_ids[i])
        assert torch.equal(data["score"], scores)

        # 只取一个样本的一个字段。
        sample = tq.kv_batch_get(
            keys=["sample_1"], partition_id=partition, select_fields=["score"]
        )
        assert set(sample.keys()) == {"score"}
        assert sample["score"].item() == 1.0
        print("sample_1 score:", sample["score"])

        tq.kv_clear(keys=keys, partition_id=partition)
        print("PASS: put / get / append field / select / clear")
    finally:
        tq.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
