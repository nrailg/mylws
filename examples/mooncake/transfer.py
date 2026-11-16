import multiprocessing as mp
import os
from multiprocessing.connection import Connection

import torch
from mooncake.engine import TransferEngine


def receive(connection: Connection) -> None:
    torch.cuda.set_device(0)
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
    tensor = torch.full((8,), -1.0, device="cuda:0")
    ptr, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
    assert engine.register_memory(ptr, size) == 0
    try:
        torch.cuda.synchronize()  # 初始化完成后才能让发送端写入。
        # Pipe 只交换地址和完成信号，Tensor 通过 Transfer Engine 传输。
        connection.send((f"127.0.0.1:{engine.get_rpc_port()}", ptr, size))
        assert connection.poll(30), "sender timed out"
        assert connection.recv() == "done"
        assert torch.equal(tensor, torch.arange(8, dtype=torch.float32, device="cuda:0"))
        print("Transfer receive:", tensor, flush=True)
    finally:
        engine.unregister_memory(ptr)
        connection.close()


def main() -> None:
    torch.cuda.set_device(0)
    # 强制使用 RDMA；spawn 子进程继承环境变量。
    assert "MC_FORCE_TCP" not in os.environ, "unset MC_FORCE_TCP to use RDMA"
    os.environ["MC_FORCE_HCA"] = "1"
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    receiver = context.Process(target=receive, args=(child,))
    receiver.start()
    child.close()
    try:
        assert parent.poll(30), "receiver timed out"
        endpoint, remote_ptr, size = parent.recv()
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
        tensor = torch.arange(8, dtype=torch.float32, device="cuda:0")
        assert tensor.numel() * tensor.element_size() == size
        ptr = tensor.data_ptr()
        assert engine.register_memory(ptr, size) == 0
        try:
            torch.cuda.synchronize()  # GPU 写完源数据后才能提交 RDMA。
            assert engine.transfer_sync_write(endpoint, ptr, remote_ptr, size) == 0
            parent.send("done")
        finally:
            engine.unregister_memory(ptr)
        receiver.join(30)
        assert receiver.exitcode == 0
        print("PASS: Transfer Engine cross-process write")
    finally:
        parent.close()
        if receiver.is_alive():
            receiver.terminate()
            receiver.join()


if __name__ == "__main__":
    main()
