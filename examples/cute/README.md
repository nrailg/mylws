# CuTe C++ 入门

CuTe 用 `Layout` 描述坐标到 offset 的映射，再用 `Tensor` 将映射应用到数据。
这组例子从手写索引开始，逐步走到 tile 和 thread partition；每个 `.cpp` 都有独立的 `main()`。

- 所有程序在 **host CPU** 上执行，不启动 CUDA kernel，也不需要 GPU。
- 编译仍需要 CUTLASS 和 CUDA Toolkit headers；这组依赖按 CUTLASS **v4.8.1** 固定。
- 每课先手算输出，再运行核对；断言通过时打印 `tXX PASS`。不要加 `-DNDEBUG`。

## 编译运行

建议在已有 CUDA Toolkit 的 Linux 开发环境运行，需要 C++17 compiler 和 Make。
CuTe 是 header-only，这些例子无需编译 CUTLASS 库或链接 CUDA runtime。

Gemini 上统一使用 `$MY_DFS/work/deps/cutlass-v4.8.1`。
当前测试服务器已准备好依赖，无需重新 clone；确认当前会话的 `MY_DFS` 后设置：

```bash
export CUTLASS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/deps/cutlass-v4.8.1"
export CUDA_HOME=/usr/local/cuda  # 改成实际 Toolkit 路径
```

仅在依赖目录尚不存在的新环境中，先准备依赖：

```bash
mkdir -p "$(dirname "$CUTLASS_DIR")"
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
```

在 mylws 根目录强制编译并运行全部例子：

```bash
cd "$MY_DFS/work/mylws"
make -B -C examples/cute check
```

产物默认放在 `outputs/cute/build/`，该目录不参与项目代码同步，避免 Mac/Linux 二进制互相覆盖。
需要独立构建目录时传入 `BUILD_DIR=/absolute/path`。
更换 `CUTLASS_DIR` 后使用 `make -B -C examples/cute check`，强制重新编译。

只学习第一课：

```bash
make -C examples/cute ../../outputs/cute/build/t01_layout
outputs/cute/build/t01_layout
```

也可以直接编译任意单个文件，观察它只依赖哪些头文件：

```bash
c++ -std=c++17 -I"$CUTLASS_DIR/include" -I"$CUDA_HOME/include" \
  examples/cute/t01_layout.cpp -o /tmp/t01_layout
/tmp/t01_layout
```

## 验证结果

2026-11-17 在 Gemini launcher 上使用默认编译选项，7 个例子全部重新编译并运行通过，
输出 `t01 PASS` 至 `t07 PASS`，退出码 0。环境：Ubuntu 22.04.5 x86_64、GCC 12.3、
CUDA Toolkit 12.8、CUTLASS v4.8.1。

复现使用上面的「编译运行」命令。

验证覆盖索引、view 共享数据、tile 坐标和 thread partition 的完整覆盖与 copy 结果；
没有启动 GPU kernel。日志：`$MY_DFS/work/mylws/outputs/cute/2026-11-17-cutlass-4.8.1/host-check.log`。

## 按顺序学习

**1. [t01_layout.cpp](t01_layout.cpp)：从 CUDA 索引到 Layout**

你熟悉的 `row * 4 + col`，对应 shape `(3,4)`、stride `(4,1)`。
`layout(row, col)` 返回元素 offset；乘上 `sizeof(T)` 才是字节 offset。
同样的 shape 配上 stride `(1,3)`，就得到 column-major 的 `row + col * 3`。

`print(layout)` 输出 `shape:stride`。本课只计算索引，没有数据数组。
先预测坐标 `(1,2)`：两种 layout 的 offset 分别是 `6` 和 `7`。
这些例子都显式指定 stride，避免把 CuTe 的默认布局误当作 row-major。

**2. [t02_static_layout.cpp](t02_static_layout.cpp)：static 与 dynamic 整数**

`3` 的类型是 `int`；`Int<3>{}` 是一个把数值 `3` 编进类型的对象。
CuTe 也提供 `_3` 这样的简写类型，写成 `_3{}` 与 `Int<3>{}` 等价。
打印出来的 `_3` 表示 static 整数，不是负数。

两种 layout 可以给出完全相同的 offset，但 static 信息能参与模板推导与编译期检查。
`auto` 让编译器推导复杂的 layout 类型；`decltype(expr)` 取得表达式的类型；
`decltype(size(static_layout))::value` 取出编译期元素数。
dynamic 类型不意味着编译器一定无法优化常量，两者区别在于类型携带的信息。

**3. [t03_hierarchical_layout.cpp](t03_hierarchical_layout.cpp)：嵌套的 shape / stride**

shape `(2,(2,3),4)` 表示两个 batch，每个有 6 行 4 列，行分成 `(inner, outer)`。
配上 stride `(24,(4,8),1)`，坐标 `(batch,(inner,outer),col)` 的 offset 为：

```text
row    = inner + 2 * outer
offset = batch * 24 + inner * 4 + outer * 8 + col = (batch * 6 + row) * 4 + col
```

顶层有 batch、行、列三个 mode，所以 `rank = 3`；tuple 嵌套两层，所以 `depth = 2`。
总元素数是 `2*2*3*4 = 48`。行坐标中 `inner` 变化最快。
先算 `(1,(1,2),3)`，应得到 `47`。程序会打印 rank 和 depth，便于对照。

**4. [t04_tensor.cpp](t04_tensor.cpp)：给 Layout 接上数据**

`make_tensor(&storage[0], layout)` 构造一个非 owning view。
`matrix(row,col)` 相当于 `storage[layout(row,col)]`；它没有复制数组，底层数组必须仍然有效。
这里显式传入首元素指针，避免模板把 C 数组本身推导成 iterator 类型。

将 shape 从 `(3,4)` 改为 `(4,3)`，stride 从 `(4,1)` 改为 `(1,4)`，得到 transpose view。
修改 `transposed(2,1)` 后，`matrix(1,2)` 和 `storage[6]` 会一起变化。
这只改变访问方式；真正将转置结果写入另一个连续数组，还需要 copy。

**5. [t05_slice.cpp](t05_slice.cpp)：固定坐标，保留 mode**

`matrix(1, _)` 固定行，留下长度为 4 的 row view；`matrix(_, 2)` 固定列，留下长度为 3 的 column view。
`_` 是 CuTe 的占位符对象，与上一课的 `_3` 类型不同。
row view 的 stride 是 `1`，column view 的 stride 是 `4`，两者继续共享原来的 storage。

预期分别读到 `4 5 6 7` 和 `2 6 10`。两个 view 在原矩阵的 `(1,2)` 处相交。

**6. [t06_local_tile.cpp](t06_local_tile.cpp)：用 tile 坐标选一个子块**

将 `4×12` 矩阵分成六个 `2×4` tile，tile grid 为 `(4/2, 12/4) = (2,3)`。
`local_tile(matrix, tile_shape, make_coord(1,1))` 返回下排中间的 tile，起点是原矩阵的 `(2,4)`：

```text
28 29 30 31
40 41 42 43
```

tile 内坐标 `(row,col)` 对应原坐标 `(tile_row*2+row, tile_col*4+col)`。
tile view 的行 stride 仍然是 `12`，不会自动压紧成 `4`，也没有搬进 shared memory。
本课 shape 恰好整除；`local_tile` 不替你检查越界或生成边界 mask。

**7. [t07_thread_partition.cpp](t07_thread_partition.cpp)：给线程分配元素**

把 8 个线程排成 `2×4`，thread layout 的 `(thread_row,thread_col)` 映射为 `thread ID`。
`local_partition(input, thread_layout, tid)` 则返回这个线程负责的数据 view。
程序在 CPU 上枚举 `tid`，便于看清完整分工。

线程排列在 `4×8` 矩阵上重复，每个线程处理 4 个位置。例如 thread 0 处理：

```text
(0,0)  (0,4)  (2,0)  (2,4)
```

thread 1 将上述列坐标各加 1。数据 layout 将坐标映射成 memory offset，
thread layout 将线程坐标映射成 thread ID；这两个 layout 的作用不同。
最后逐个核对输出与输入相同，并检查每个位置恰好被访问一次。
这里没有验证真实线程并发、同步或访存性能。

## 后续衔接

能手算第 7 课的分工后，再把这种 partition 放进 CUDA copy kernel，用 `threadIdx.x` 提供 `tid`。
接下来学习 reduction 和基础 GEMM，然后才需要 `TiledCopy`、`TiledMMA` 与异步 pipeline。

参考：[CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)、
[Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)、
[官方 C++ 示例](https://github.com/NVIDIA/cutlass/tree/v4.8.1/examples/cute/tutorial)。
