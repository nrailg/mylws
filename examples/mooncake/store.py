import os

import torch
from mooncake.store import MooncakeDistributedStore


def main() -> None:
    os.environ["MC_FORCE_TCP"] = "1"
    os.environ["MC_TCP_BIND_ADDRESS"] = "127.0.0.1"
    store = MooncakeDistributedStore()
    try:
        # 贡献 64 MiB 存储内存；16 MiB 是客户端传输 buffer。
        assert store.setup(
            "127.0.0.1", "P2PHANDSHAKE", 64 << 20, 16 << 20,
            "tcp", "", "127.0.0.1:50051",
        ) == 0

        tensor = torch.arange(8, dtype=torch.float32)
        ptr, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        assert store.register_buffer(ptr, size) == 0
        try:
            # Store 保存原始字节；shape / dtype 由调用方掌握。
            assert store.put_from("quickstart/tensor", ptr, size) == 0
            tensor.fill_(-1)
            assert store.get_into("quickstart/tensor", ptr, size) == size
            assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))
            print("Store get:", tensor)
            # 已完成读取，强制清理本例的 key，跳过读取租约保护。
            assert store.remove("quickstart/tensor", force=True) == 0
            assert store.is_exist("quickstart/tensor") == 0
            print("PASS: Store put / get / remove")
        finally:
            store.unregister_buffer(ptr)
    finally:
        store.close()


if __name__ == "__main__":
    main()
