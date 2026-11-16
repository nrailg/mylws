#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    // Layout 只计算 offset，不分配或访问数据；stride 的单位是元素。
    auto row_major = make_layout(make_shape(3, 4), make_stride(4, 1));
    auto column_major = make_layout(make_shape(3, 4), make_stride(1, 3));

    std::printf("row_major: ");
    print(row_major);
    std::printf("\n(row, col) -> row-major offset / column-major offset\n");

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            int row_offset = row_major(row, col);
            int column_offset = column_major(row, col);
            assert(row_offset == row * 4 + col);
            assert(column_offset == row + col * 3);
            std::printf("(%d,%d) -> %2d / %2d  ", row, col, row_offset, column_offset);
        }
        std::printf("\n");
    }

    assert(rank(row_major) == 2);
    assert(size(row_major) == 12);
    std::puts("t01 PASS");
}
