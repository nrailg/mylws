import multiprocessing as mp
import os
from multiprocessing.connection import Connection

import torch
from mooncake.engine import TransferEngine


def receive(connection: Connection) -> None:
    engine = TransferEngine()
    assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
    tensor = torch.full((8,), -1.0)
    ptr, size = tensor.data_ptr(), tensor.numel() * tensor.element_size()
    assert engine.register_memory(ptr, size) == 0
    try:
        # Pipe 只交换地址和完成信号，Tensor 通过 Transfer Engine 传输。
        connection.send((f"127.0.0.1:{engine.get_rpc_port()}", ptr, size))
        assert connection.poll(30), "sender timed out"
        assert connection.recv() == "done"
        assert torch.equal(tensor, torch.arange(8, dtype=torch.float32))
        print("Transfer receive:", tensor, flush=True)
    finally:
        engine.unregister_memory(ptr)
        connection.close()


def main() -> None:
    # 显式关闭 transport 自动选择；spawn 子进程继承环境变量。
    os.environ["MC_FORCE_TCP"] = "1"
    os.environ["MC_TCP_BIND_ADDRESS"] = "127.0.0.1"
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    receiver = context.Process(target=receive, args=(child,))
    receiver.start()
    child.close()
    try:
        assert parent.poll(30), "receiver timed out"
        endpoint, remote_ptr, size = parent.recv()
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        tensor = torch.arange(8, dtype=torch.float32)
        assert tensor.numel() * tensor.element_size() == size
        ptr = tensor.data_ptr()
        assert engine.register_memory(ptr, size) == 0
        try:
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
