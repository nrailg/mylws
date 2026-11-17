#include <cassert>
#include <cstdio>

#include <cute/tensor.hpp>

int main() {
    using namespace cute;

    int source[32];
    int destination[32] = {};
    int visits[32] = {};
    for (int i = 0; i < 32; ++i) {
        source[i] = 100 + i;
    }
    auto data_layout = make_layout(make_shape(4, 8), make_stride(8, 1));
    auto input = make_tensor(&source[0], data_layout);
    auto output = make_tensor(&destination[0], data_layout);

    // thread_layout 将 (thread_row, thread_col) 映射为 thread ID，不是数据 offset。
    auto thread_layout = make_layout(make_shape(Int<2>{}, Int<4>{}),
                                     make_stride(Int<4>{}, Int<1>{}));
    for (int tid = 0; tid < size(thread_layout); ++tid) {
        // 在 host 上枚举 thread ID，展示分工；这里没有启动 CUDA 线程。
        /* md
         * 对比原矩阵中的逻辑坐标：两种方式都取 `(2, 2)` 的 view。
         * L：`local_tile` 用 tile shape `(2,2)`、tile 坐标 `(0,0)`。
         * P：`local_partition` 用当前 thread layout `(2,4)`、`tid=0`。
         * $$
         * \small
         * \begin{aligned}
         *   I &= \begin{bmatrix}
         *     \mathbf{100} & 101 & 102 & 103 & \mathbf{104} & 105 & 106 & 107 \\
         *     108          & 109 & 110 & 111 & 112          & 113 & 114 & 115 \\
         *     \mathbf{116} & 117 & 118 & 119 & \mathbf{120} & 121 & 122 & 123 \\
         *     124          & 125 & 126 & 127 & 128          & 129 & 130 & 131
         *   \end{bmatrix} \\[6pt]
         *   L(i,j) &= I(i,j),\quad
         *   P(i,j) = I(2i,4j),\quad i,j\in\{0,1\} \\[4pt]
         *   L &= \begin{bmatrix}
         *     100 & 101 \\
         *     108 & 109
         *   \end{bmatrix},\qquad
         *   P = \begin{bmatrix}
         *     100 & 104 \\
         *     116 & 120
         *   \end{bmatrix}
         * \end{aligned}
         * $$
         * L 取相邻行列；P 每隔 2 行、4 列取值（原矩阵中已加粗）。两者都不复制数据。
         */
        auto thread_input = local_partition(input, thread_layout, tid);
        auto thread_output = local_partition(output, thread_layout, tid);
        assert(size<0>(thread_input) == 2);
        assert(size<1>(thread_input) == 2);

        int thread_row = tid / 4;
        int thread_col = tid % 4;
        assert(thread_layout(thread_row, thread_col) == tid);
        std::printf("thread %d owns:", tid);
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 2; ++col) {
                int global_row = thread_row + row * 2;
                int global_col = thread_col + col * 4;
                int offset = global_row * 8 + global_col;
                assert(thread_input(row, col) == source[offset]);
                thread_output(row, col) = thread_input(row, col);
                ++visits[offset];
                std::printf(" (%d,%d)", global_row, global_col);
            }
        }
        std::printf("\n");
    }

    for (int i = 0; i < 32; ++i) {
        assert(visits[i] == 1);
        assert(destination[i] == source[i]);
    }
    std::puts("t07 PASS");
}
