#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    printf("------------------------------------------------------\n");
    /* md
     * **complement：算出各个 tile 应放在哪些起点，才能覆盖目标范围**
     *
     * 例如要覆盖位置 `0..5`，每个 tile 取相对起点的 `{0,1}` 两个位置。
     * 应该从哪些位置开始取？`complement` 帮你算出起点 `{0,2,4}`：
     *
     * | tile 起点 | 加上 tile 内的相对 offset | 覆盖的位置 |
     * | --- | --- | --- |
     * | 0 | `0 + {0,1}` | `{0,1}` |
     * | 2 | `2 + {0,1}` | `{2,3}` |
     * | 4 | `4 + {0,1}` | `{4,5}` |
     *
     * 对应下面的三行代码：
     *
     * - `tile = make_layout(Int<2>{}, Int<1>{})`：定义一个 tile 的相对 offset。
     *   shape 为 2、stride 为 1，输入 i 为 0、1 时，`tile(i)` 输出 0、1。
     * - `rest = complement(tile, Int<6>{})`：求覆盖 `0..5` 所需的 tile 起点。
     *   返回 Layout `3:2`，输入 t 为 0、1、2 时，`rest(t)` 输出 0、2、4。
     * - `combined = make_layout(tile, rest)`：给定 tile 编号 t 和其中的元素编号 i，
     *   算出该元素的位置：`combined(i,t) = tile(i) + rest(t) = i + 2*t`。
     *   例如第 2 个 tile 的第 1 个元素位于 `combined(1,2) = 1 + 4 = 5`（编号从 0 开始）。
     *
     * 这里 `make_layout` 接受两个 Layout，分别取它们的 shape 和 stride：
     *
     * | Layout | shape | stride |
     * | --- | --- | --- |
     * | `tile` | `2` | `1` |
     * | `rest` | `3` | `2` |
     * | `combined` | `(2,3)` | `(1,2)` |
     *
     * `i`、`t` 是逻辑坐标，计算结果是 offset，不是 buffer 中的数据。
     * 本课使用可整除的 static 参数，不讨论越界或 padding。
     */
    // contiguous tile: _2:_1; complement: _3:_2; combined: (_2,_3):(_1,_2)
    // tile 0: 0 1
    // tile 1: 2 3
    // tile 2: 4 5
    auto tile = make_layout(Int<2>{}, Int<1>{});
    auto rest = complement(tile, Int<6>{});
    auto combined = make_layout(tile, rest);
    std::printf("contiguous tile: ");
    print(tile);
    std::printf("; complement: ");
    print(rest);
    std::printf("; combined: ");
    print(combined);
    std::printf("\n");
    assert(size(rest) == 3);
    assert(stride(rest) == 2);
    int visits[6] = {};
    for (int t = 0; t < 3; ++t) {
        std::printf("tile %d:", t);
        for (int i = 0; i < 2; ++i) {
            int k = combined(i, t);
            assert(k == tile(i) + rest(t));
            assert(k == 2 * t + i);
            assert(k >= 0 && k < 6);
            ++visits[k];
            std::printf(" %d", k);
        }
        std::printf("\n");
    }
    for (int k = 0; k < 6; ++k) {
        assert(visits[k] == 1);
    }

    printf("------------------------------------------------------\n");
    /* md
     * **直接把二维 target `(8,9)` 传给 complement，并使用返回的 Layout**
     *
     * `tile_2d = (2,3):(3,1)` 的两行 offset 是 `{0,1,2}`、`{3,4,5}`。
     * `rest_2d = complement(tile_2d, target_shape)` 在 CUTLASS v4.8.1 返回 `18:6`。
     * 后面的 `combined_2d` 和遍历都使用这个结果。
     *
     * 为什么是 18 个起点，而不是 `72/6=12` 个？内部对 tuple 逐项向上取整：
     *
     * | 步骤 | 计算 | 结果 |
     * | --- | --- | --- |
     * | target 第 0 个 mode | `ceil(8/6)` | `2` |
     * | 剩余除数 | `ceil(6/8)` | `1` |
     * | target 第 1 个 mode | `ceil(9/1)` | `9` |
     *
     * `ceil_div((8,9),6)` 得到 `(2,9)`，生成 stride `(6,12)`，合并为 `18:6`。
     *
     * | Layout | shape | stride |
     * | --- | --- | --- |
     * | `tile_2d` | `(2,3)` | `(3,1)` |
     * | `rest_2d` | `18` | `6` |
     * | `combined_2d` | `((2,3),18)` | `((3,1),6)` |
     *
     * `combined_2d(make_coord(row,col), t) = 3*row + col + 6*t`。
     * 18 个 tile 覆盖 `0..107`：前 12 个覆盖目标范围 `0..71`，后 6 个覆盖 `72..107`。
     * 程序遍历全部 18 个 tile，验证每个 offset 恰好出现一次，并标出目标范围外的位置。
     * 这里只计算 offset；若实际 buffer 只有 72 个元素，不能访问范围外的 36 个位置。
     *
     * 直接传 tuple 不保证只覆盖其元素乘积大小，也不代表按 8 行 9 列切矩形子块。
     * 本例保留这个实际行为；传 `size(target_shape)` 得到 `12:6` 是另一种调用。
     */
    auto tile_2d = make_layout(make_shape(Int<2>{}, Int<3>{}),
                               make_stride(Int<3>{}, Int<1>{}));
    auto target_shape = make_shape(Int<8>{}, Int<9>{});
    auto rest_2d = complement(tile_2d, target_shape);
    auto combined_2d = make_layout(tile_2d, rest_2d);
    std::printf("2D tile: ");
    print(tile_2d);
    std::printf("; target shape: ");
    print(target_shape);
    std::printf("; complement: ");
    print(rest_2d);
    std::printf("; combined: ");
    print(combined_2d);
    std::printf("\n");
    assert(size(rest_2d) == 18);
    assert(stride(rest_2d) == 6);
    assert(size(combined_2d) == 108);
    int visits_2d[108] = {};
    int target_count = 0;
    int extra_count = 0;
    for (int t = 0; t < size(rest_2d); ++t) {
        std::printf("tile %2d, start %3d:\n", t, int(rest_2d(t)));
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 3; ++col) {
                int k = combined_2d(make_coord(row, col), t);
                assert(k == tile_2d(row, col) + rest_2d(t));
                assert(k == 6 * t + 3 * row + col);
                assert(k >= 0 && k < 108);
                ++visits_2d[k];
                bool in_target = k < size(target_shape);
                if (in_target) {
                    ++target_count;
                } else {
                    ++extra_count;
                }
                std::printf(" %3d%s", k, in_target ? "" : "(extra)");
            }
            std::printf("\n");
        }
    }
    for (int k = 0; k < 108; ++k) {
        assert(visits_2d[k] == 1);
    }
    assert(target_count == 72);
    assert(extra_count == 36);
    std::printf("target offsets: %d; extra offsets: %d\n", target_count, extra_count);

    printf("------------------------------------------------------\n");
    /* md
     * **换成间隔取值：complement 也会改变**
     *
     * `B = 2:3` 取 `{0,3}`，`complement(B, 6)` 得到 `Q = 3:1`。
     * 三个位移 `{0,1,2}` 分别补出 `{0,3}`、`{1,4}`、`{2,5}`。
     * 合起来仍覆盖 `0..5`，但分组方式不同：$K(i,t)=3i+t$。
     *
     * **logical_divide：把分组规则接到原数据 layout 上**
     *
     * 令原 layout `A = 6:2`，即 $A(k)=2k$。k 是逻辑坐标，最终 offset 还要经过 A。
     * `logical_divide(A, B)` 等价于本例下面两步 `make_layout` 加 `composition`：
     *
     * $$
     * \small
     * \begin{aligned}
     *   Q &= \operatorname{complement}(B,6)=3:1 \\
     *   K &= (B,Q)=(2,3):(3,1) \\
     *   D(i,t) &= A(K(i,t))=2(3i+t), & D &= (2,3):(6,2)
     * \end{aligned}
     * $$
     *
     * | tile 坐标 t | 原逻辑坐标 k | 最终 offset A(k) |
     * | --- | --- | --- |
     * | 0 | `{0,3}` | `{0,6}` |
     * | 1 | `{1,4}` | `{2,8}` |
     * | 2 | `{2,5}` | `{4,10}` |
     *
     * D 的第 0 个 mode 是 tile 内坐标 i，第 1 个 mode 是 tile 坐标 t。
     * 这些操作都返回 Layout，不搬运数据。下一步才把分组推广到二维矩阵。
     */
    // strided complement: _3:_1; logical_divide: (_2,_3):(_6,_2)
    // (i,t) -> logical coordinate -> offset
    // (0,0) -> 0 ->  0  (1,0) -> 3 ->  6
    // (0,1) -> 1 ->  2  (1,1) -> 4 ->  8
    // (0,2) -> 2 ->  4  (1,2) -> 5 -> 10
    // t09 PASS
    auto strided_tile = make_layout(Int<2>{}, Int<3>{});
    auto strided_rest = complement(strided_tile, Int<6>{});
    auto strided_combined = make_layout(strided_tile, strided_rest);
    auto original = make_layout(Int<6>{}, Int<2>{});
    auto manual = composition(original, strided_combined);
    auto divided = logical_divide(original, strided_tile);
    assert(size(strided_rest) == 3);
    assert(stride(strided_rest) == 1);
    assert(size<0>(divided) == 2);
    assert(size<1>(divided) == 3);
    assert(stride<0>(divided) == 6);
    assert(stride<1>(divided) == 2);
    std::printf("strided complement: ");
    print(strided_rest);
    std::printf("; logical_divide: ");
    print(divided);
    std::printf("\n(i,t) -> logical coordinate -> offset\n");
    int strided_visits[6] = {};
    for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < 2; ++i) {
            int k = strided_combined(i, t);
            int offset = divided(i, t);
            assert(k == 3 * i + t);
            assert(k >= 0 && k < 6);
            ++strided_visits[k];
            assert(offset == manual(i, t));
            assert(offset == original(k));
            assert(offset == 6 * i + 2 * t);
            std::printf("(%d,%d) -> %d -> %2d  ", i, t, k, offset);
        }
        std::printf("\n");
    }
    for (int k = 0; k < 6; ++k) {
        assert(strided_visits[k] == 1);
    }
    std::puts("t09 PASS");
}
