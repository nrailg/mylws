# CuTe C++ 入门

CuTe 用 `Layout` 描述坐标到 offset 的映射，再用 `Tensor` 将映射应用到数据。
这组例子从手写索引开始，逐步走到 tile、thread partition 和 layout algebra；每个 `.cpp` 都有独立的 `main()`。

- `t01` 到 `t09` 在 **host CPU** 上执行。[sgemm_1](#sgemm_1cuda-gemm) 需要一张 CUDA GPU，[wgmma_tma_sm90](#wgmma_tma_sm90) 需要一张 Hopper GPU。
- 编译仍需要 CUTLASS 和 CUDA Toolkit headers；这组依赖按 CUTLASS **v4.8.1** 固定。
- CPU 例子先手算输出，再运行核对。断言通过时打印 `tXX PASS`。不要加 `-DNDEBUG`。

## 编译运行

建议在已有 CUDA Toolkit 的 Linux 开发环境运行，需要 C++17 compiler 和 Make。
CuTe 是 header-only。CPU 例子用到 CUTLASS 和 CUDA headers。

Gemini 上统一使用 `$MY_DFS/work/deps/cutlass-v4.8.1`。
确认当前会话的 `MY_DFS` 后设置依赖路径。

```bash
export CUTLASS_DIR="${MY_DFS:?set MY_DFS from the current Gemini session}/work/deps/cutlass-v4.8.1"
export CUDA_HOME=/usr/local/cuda  # 改成实际 Toolkit 路径
```

仅在依赖目录尚不存在的新环境中，先准备依赖：

```bash
mkdir -p "$(dirname "$CUTLASS_DIR")"
git clone --depth 1 --branch v4.8.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
```

在 mylws 根目录增量编译并运行全部 CPU 例子。

```bash
cd "$MY_DFS/work/mylws"
make -C examples/cute check
```

产物默认放在 `outputs/cute/build/`，该目录不参与项目代码同步，避免 Mac/Linux 二进制互相覆盖。
未改动的 `.cpp` 会跳过编译；`check` 每次仍会运行全部例子。只编译时使用 `make -C examples/cute`。
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

2026-11-17 在 Gemini launcher 上使用默认编译选项，9 个例子全部强制重新编译并运行通过。
随后将 t09 改为直接使用 tuple target 的返回值，又单独重编译运行通过：
`target offsets: 72; extra offsets: 36`、`t09 PASS`，退出码 0。
补充 t08 的不可合并案例后，单独重编译运行也通过：rank 保持 2，全部 offset 不变，`t08 PASS`。
环境：Ubuntu 22.04.5 x86_64、GCC 12.3、
CUDA Toolkit 12.8、CUTLASS v4.8.1。

复现使用上面的「编译运行」命令。

验证覆盖索引、view 共享数据、tile/线程分工、layout algebra，以及 complement 的 tuple/size 参数差异。
没有启动 GPU kernel。最新 t09 日志：`$MY_DFS/work/mylws/outputs/cute/2026-11-17-complement-fix/tuple-target-check.log`。
t08 不可合并案例日志：`$MY_DFS/work/mylws/outputs/cute/2026-11-17-coalesce-check/host-check.log`。

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

**8. [t08_layout_algebra.cpp](t08_layout_algebra.cpp)：coalesce 与 composition**

先区分一维逻辑坐标与 offset：shape 为 `(m,n)` 时，整数 `k` 对应 `(k % m, k / m)`，
再根据 stride 算 offset。这个展开顺序与 row-major/column-major 的 stride 无关。

- `coalesce(A)` 简化 layout，保证 `size` 不变且每个合法整数 `k` 的映射不变。
  本课 `(2,3):(1,2)` 可合并为 `6:1`；`(2,3):(3,1)` 不能合并成一个 mode，
  因为它按 `k=0..5` 访问的 offset 是 `0,3,1,4,2,5`。使用 static 参数展示编译期合并。
  不可合并时正常返回，本例 shape/stride 保持不变、rank 仍为 2；程序打印调用前后结果并逐项断言。
- `composition(A,B)` 构造 `R(c)=A(B(c))`。本课 `A=(3,4):(4,1)`、`B=(3,2):(1,6)`，
  得到 `R=(3,2):(4,2)`，选中 A 的第 0、2 列。B 的输出是 A 的一维逻辑坐标。

先手算 `B(1,1)=7`，再算 `A(7)=A(1,2)=6`；不能把 `A(7)` 当成 offset 7。
两种操作都只构造映射，不复制数据；`coalesce` 也不保证任意 layout 都能降成 rank 1。

在已配置 CUTLASS/CUDA headers 的 Linux 环境运行：

```bash
make -C examples/cute ../../outputs/cute/build/t08_layout_algebra
outputs/cute/build/t08_layout_algebra
```

**9. [t09_complement_divide.cpp](t09_complement_divide.cpp)：complement 与 logical_divide**

先固定目标逻辑坐标 `0..5`，比较两种分组；所有参数均为 static：

| tile layout B | complement(B, 6) | 三个 tile 各自覆盖的逻辑坐标 |
| --- | --- | --- |
| `2:1` | `3:2` | `{0,1}`、`{2,3}`、`{4,5}` |
| `2:3` | `3:1` | `{0,3}`、`{1,4}`、`{2,5}` |

complement 描述补齐覆盖所需的位移，不是简单的集合差。
`make_layout(B, complement(B, 6))` 构造 `(tile 内坐标, tile 坐标)` 到逻辑坐标的映射。

中间的二维输入例子直接调用 `complement(B, target_shape)`，其中 `B=(2,3):(3,1)`、
target shape 为 `(8,9)`。CUTLASS v4.8.1 内部 `ceil_div((8,9),6)` 得到 `(2,9)`，
返回 `18:6`；程序使用这个结果构造 `((2,3),18):((3,1),6)`，遍历全部 18 个 tile。
验证 `0..107` 恰好覆盖一次，其中目标范围 `0..71` 有 72 个位置，额外范围 `72..107` 有 36 个位置，
输出以 `(extra)` 标注。只计算 offset，不访问实际 buffer。

这不等于在 8×9 矩阵上切 2×3 的矩形子块；tuple 的逐项向上取整可能带来额外覆盖。
传 `size(target_shape)` 得到 `12:6` 是另一种调用，本例不以它替换 tuple target。

再接上原数据 layout `A=6:2`：`logical_divide(A,B)` 等价于将上述组合与 A 做 composition。
对于 `B=2:3`，结果是 `(2,3):(6,2)`；例如 `(i,t)=(1,2)` 先映射到逻辑坐标 `5`，
再得到 offset `10`。程序验证两种分组都恰好覆盖 `0..5`，并对照手动组合与 `logical_divide`。
本课没有复制数据；二维 target 示例明确检查额外覆盖，其余两个一维示例恰好覆盖目标范围。

```bash
make -C examples/cute ../../outputs/cute/build/t09_complement_divide
outputs/cute/build/t09_complement_divide
```

## sgemm_1（CUDA GEMM）

[sgemm_1.cu](sgemm_1.cu) 沿用 CUTLASS v4.8.1 教程中的 `gemm_device`、`gemm_nt` 和
`gemm_tn`。每个 CTA 用 `local_tile` 选取数据，线程按 `tA` 和 `tB` 把 A、B 搬到
shared memory，再按 `tC` 分工调用 `cute::gemm` 累加，最后用 `axpby` 写回 C。
参考 [NVIDIA CuTe GEMM 教程](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html#sgemm-1-cu)。

沿用上面的 `CUTLASS_DIR` 和 `CUDA_HOME`，在有 `nvcc` 和一张 CUDA GPU 的 Linux 环境运行。

```bash
make -C examples/cute check-gpu
```

默认用 `CUDA_ARCH=native`。需要指定架构时可传入 `CUDA_ARCH=sm_90`，切换架构后加 `-B`
重新编译。教程 kernel 按完整的 `(128, 128, 8)` tile 读写，demo 固定使用对齐的
`(M, N, K) = (256, 384, 24)`。程序默认在 GPU 0 上运行 NT，也可取消注释
`main()` 中的 `check_gemm<false>()` 来运行 TN。两种布局都用 CPU 逐元素核对
`C = alpha * A * B^T + beta * C`，其中 CuTe 的 B 逻辑 shape 是 `(N, K)`。

2026-11-21 在单节点 Gemini launcher 的 NVIDIA H20 上用 CUDA 12.8.93、GCC 12.3 和
CUTLASS v4.8.1 运行 `check-gpu CUDA_ARCH=sm_90`。在临时测试副本中启用 TN 后，
`sgemm_1 NT PASS` 和 `sgemm_1 TN PASS` 均通过，退出码为 0。
日志位于 Gemini 项目目录的 `outputs/cute/20261121-sgemm-select/check-gpu.log`。

## wgmma_tma_sm90

[wgmma_tma_sm90.cu](wgmma_tma_sm90.cu) 用 TMA 把 A 和 B 异步搬到 shared memory 中的三段
pipeline，warpgroup 等待对应的 barrier 后用 WGMMA 累加 FP16 GEMM。源自
[NVIDIA CUTLASS v4.2.1 的 Hopper 教程](https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu)。

沿用上面的 `CUTLASS_DIR` 和 `CUDA_HOME`，在一张 Hopper GPU（如 H20 或 H100）上运行。
Makefile 以 `sm_90a` 编译此例。

```bash
make -C examples/cute check-wgmma-tma
```

命令用 `(M, N, K) = (512, 256, 1024)` 分别运行 NT 和 TN，逐元素核对 CPU reference，
每种布局通过时打印 `wgmma_tma_sm90 ... PASS`，随后打印计时。
单独编译可用 `make -C examples/cute ../../outputs/cute/build/wgmma_tma_sm90`。
可执行文件接受 `M N K N T` 或 `M N K T N`。完整 CTA tile 和 cluster 要求
`M`、`N`、`K` 分别为 256、128、64 的正整数倍。输入为 ±1，`K <= 2048` 时做精确校验。

2026-11-23 在单节点 Gemini 的 NVIDIA H20 上用 CUDA 12.8.93、GCC 12.3 和 CUTLASS v4.8.1
运行 `check-wgmma-tma`。默认形状和 `(256, 128, 64)` 的 NT、TN 逐元素校验均通过，
退出码为 0。未对齐的 `M=128` 返回退出码 1。
日志位于 Gemini 项目目录的 `outputs/cute/20261123-wgmma-tma.954Uam/check-wgmma-tma.log`。

## 后续衔接

接下来将 divide 推广到二维，比较 logical/zipped/tiled/flat divide，再补 layout product 与 thread-value mapping，进入真实 GPU GEMM、
Tensor Core MMA 和异步 pipeline。当前 host 示例还不足以覆盖 DeepGEMM 的执行机制。

参考：[CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)、
[Layout Algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html)、
[Tensor](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)、
[官方 C++ 示例](https://github.com/NVIDIA/cutlass/tree/v4.8.1/examples/cute/tutorial)。
