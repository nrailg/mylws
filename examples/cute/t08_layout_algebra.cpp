#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    /* md
     * `coalesce` 用来化简 layout，让后续索引计算处理更少的 mode。
     * 例如 shape `(2,3)` 把整数 k 展开为 `(k % 2, k / 2)`，再用 stride 算 offset：
     * $$
     * \small
     * \begin{aligned}
     *   C &= (2,3):(1,2) \\
     *   C(k) &= (k\bmod2)+2\lfloor k/2\rfloor=k \\
     *   \operatorname{coalesce}(C) &= 6:1,\qquad 0\le k<6
     * \end{aligned}
     * $$
     * 化简后直接用 `k * 1` 表达映射，不再需要两个 mode 的分解与组合。
     * 它可整理切片、组合等操作产生的复杂 layout；本例用于展示化简，并非必须调用。
     * 保持每个 `C(k)` 的结果，但不保留原二维 shape，也不搬运数据。
     * 这里用 static 整数让合并条件 `2 * 1 == 2` 在编译期可知；不保证运行更快。
     */
    auto column_major = make_layout(make_shape(Int<2>{}, Int<3>{}),
                                    make_stride(Int<1>{}, Int<2>{}));
    auto merged = coalesce(column_major);
    std::printf("coalesce: ");
    print(column_major);
    std::printf(" -> ");
    print(merged);
    std::printf("\n");
    assert(rank(merged) == 1);
    assert(size(merged) == 6);
    assert(stride(merged) == 1);
    for (int k = 0; k < 6; ++k) {
        assert(column_major(k) == k);
        assert(merged(k) == column_major(k));
    }

    /* md
     * **不能合并时，调用 coalesce 会怎样？**
     *
     * `row_major = (2,3):(3,1)` 不满足相邻 mode 的合并条件：`2 * 3 != 1`。
     * `coalesce(row_major)` 正常返回，仍为 `(2,3):(3,1)`，rank 仍为 2；不会报错。
     * 输入 k 为 `0..5` 时，调用前后的 offset 都是 `0,3,1,4,2,5`。
     * 这个序列无法用一个固定 stride 表示，所以需要保留两个 mode。
     *
     * 更复杂的 layout 可以只合并其中满足条件的部分；coalesce 不保证结果 rank 为 1。
     */
    auto row_major = make_layout(make_shape(Int<2>{}, Int<3>{}),
                                 make_stride(Int<3>{}, Int<1>{}));
    auto unmerged = coalesce(row_major);
    assert(rank(unmerged) == 2);
    assert(shape(unmerged) == shape(row_major));
    assert(stride(unmerged) == stride(row_major));
    std::printf("coalesce (cannot merge): ");
    print(row_major);
    std::printf(" -> ");
    print(unmerged);
    std::printf("\nrank: %d -> %d\n", int(rank(row_major)), int(rank(unmerged)));
    std::printf("k | before | after\n");
    for (int k = 0; k < 6; ++k) {
        int row = k % 2;
        int col = k / 2;
        assert(row_major(k) == row * 3 + col);
        assert(unmerged(k) == row_major(k));
        std::printf("%d | %6d | %5d\n", k, int(row_major(k)), int(unmerged(k)));
    }
    std::printf("\n");

    /* md
     * **composition：把坐标变换组合成新的 Layout**
     *
     * `auto composed = composition(a, b)` 得到 Layout R，满足 $R(r,c)=A(B(r,c))$。
     * 它只有坐标映射，可传给 `make_tensor(ptr, composed)` 构造共享 buffer 的 view。
     *
     * - B 描述选取或重排规则，A 将选中的逻辑坐标映射到 offset。
     * - 本例选第 0、2 列，类似 slice；也可继续组合，实现多层选取。
     * - 还可表达元素重排，以及 `(thread, value)` 到数据 offset 的映射。
     * - R 的输入来自 B；$A(B(C(D(x))))$ 的输入来自 D。
     *   合法复合保持输入坐标兼容，但可能拆分 mode，shape 结构未必相同。
     *
     * **标量逻辑坐标如何展开**
     *
     * shape 为 $(d_A,d_B,d_C)$，$0\le k<d_A d_B d_C$。第 0 个 mode 先变化，
     * 满了向下一 mode 进位；展开过程不需要 stride：
     *
     * $$
     * \small
     * \begin{aligned}
     *   a &= k\bmod d_A, &
     *   b &= \lfloor k/d_A\rfloor\bmod d_B, &
     *   c &= \lfloor k/(d_A d_B)\rfloor \\
     *   k &= a+d_A b+d_A d_B c
     * \end{aligned}
     * $$
     *
     * 例如 shape `(2,3,4)` 将 `k=17` 展开为 `(1,2,2)`。
     * 随后才用 stride 计算 $\mathrm{offset}=a s_A+b s_B+c s_C$。
     * shape 相同就展开成相同坐标；stride 不同，offset 可能不同。
     * **从 memory offset 反推坐标**则需要考虑 stride，是另一个问题。
     *
     * **本例：选第 0、2 列**
     *
     * | Layout | shape | stride |
     * | --- | --- | --- |
     * | A | `(3,4)` | `(4,1)` |
     * | B | `(3,2)` | `(1,6)` |
     * | R | `(3,2)` | `(4,2)` |
     *
     * 先由 B 得到 k，再由 A 展开为 `(k % 3, k / 3)` 并计算 offset。
     * 例：$B(1,1)=7$，$A(7)=A(1,2)=6$，所以 `composed(1,1) == 6`。
     * 下列矩阵列出 **offset**，不是 buffer 中的数据：
     *
     * $$
     * \small
     * A = \begin{bmatrix}
     *   0 & 1 &  2 &  3 \\
     *   4 & 5 &  6 &  7 \\
     *   8 & 9 & 10 & 11
     * \end{bmatrix},\qquad
     * R = \begin{bmatrix}
     *   0 &  2 \\
     *   4 &  6 \\
     *   8 & 10
     * \end{bmatrix}
     * $$
     */
    auto a = make_layout(make_shape(Int<3>{}, Int<4>{}),
                         make_stride(Int<4>{}, Int<1>{}));
    auto b = make_layout(make_shape(Int<3>{}, Int<2>{}),
                         make_stride(Int<1>{}, Int<6>{}));
    auto composed = composition(a, b);
    std::printf("composition(A, B): ");
    print(composed);
    std::printf("\n(r,c) -> B(r,c) -> A(B(r,c))\n");
    assert(size<0>(composed) == 3);
    assert(size<1>(composed) == 2);
    assert(stride<0>(composed) == 4);
    assert(stride<1>(composed) == 2);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 2; ++col) {
            int k = b(row, col);
            int offset = composed(row, col);
            assert(offset == a(k));
            assert(offset == a(row, 2 * col));
            assert(offset == row * 4 + 2 * col);
            std::printf("(%d,%d) -> %d -> %2d  ", row, col, k, offset);
        }
        std::printf("\n");
    }
    std::puts("t08 PASS");
}
