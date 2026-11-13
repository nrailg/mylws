"""TensorDict quickstart：在 mylws 根目录执行 .venv/bin/python examples/tq/tensor_dict_demo.py。"""

import torch
from tensordict import TensorDict


def main() -> None:
    # 1. 两个样本，共享 batch 维 [2]；各字段的特征维、dtype 可以不同。
    batch = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "metrics": {"score": torch.tensor([0.5, 1.0])},
        },
        batch_size=[2],
    )
    print("1. batch_size:", list(batch.batch_size))
    print("input_ids shape:", list(batch["input_ids"].shape))
    print("nested score:", batch["metrics", "score"].tolist())
    # 新字段也必须匹配 batch 维；这里每个样本有一个长度值。
    batch["length"] = batch["attention_mask"].sum(-1)
    assert batch["length"].tolist() == [2, 3]

    # 2. 字符串取字段；整数、切片、布尔 mask 同时索引所有字段。
    sample = batch[0]
    selected = batch[batch["metrics", "score"] > 0.5]
    fields = batch.select("input_ids", "length")
    print("2. sample[0]:", sample["input_ids"].tolist(), "length:", sample["length"].item())
    print("score > 0.5:", selected["input_ids"].tolist())
    print("select fields:", list(fields.keys()))
    assert sample.batch_size == torch.Size([])
    assert selected.batch_size == torch.Size([1])
    assert selected["length"].tolist() == [3]
    assert set(fields.keys()) == {"input_ids", "length"}

    # 普通切片共享底层 Tensor；需要独立修改时先 clone()。
    independent = batch[:1].clone()
    independent["input_ids"].zero_()
    assert batch["input_ids"][0].tolist() == [1, 2, 3]

    # 3. cat 沿已有 batch 维拼接；stack 新增 batch 维，嵌套字段一起处理。
    joined = torch.cat([batch, batch], dim=0)
    stacked = torch.stack([batch, batch], dim=0)
    flat = stacked.reshape(-1)
    moved = batch.to("cpu")  # GPU 环境可改为 "cuda"，所有字段统一迁移。
    doubled = batch.select("metrics").apply(lambda tensor: tensor * 2)
    print("3. cat batch_size:", list(joined.batch_size))
    print("stack batch_size:", list(stacked.batch_size))
    print("stack input_ids shape:", list(stacked["input_ids"].shape))
    print("reshape batch_size:", list(flat.batch_size))
    print("to device:", moved.device)
    print("apply score * 2:", doubled["metrics", "score"].tolist())
    assert joined.batch_size == torch.Size([4])
    assert stacked.batch_size == torch.Size([2, 2])
    assert stacked["input_ids"].shape == torch.Size([2, 2, 3])
    assert torch.equal(flat["input_ids"], joined["input_ids"])
    assert torch.equal(flat["metrics", "score"], joined["metrics", "score"])
    assert moved["input_ids"].device.type == "cpu"
    assert doubled["metrics", "score"].tolist() == [1.0, 2.0]
    print("PASS: create / nested keys / index / select / clone / cat / stack / reshape / to / apply")


if __name__ == "__main__":
    main()
