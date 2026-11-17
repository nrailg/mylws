#include <cassert>
#include <cstdio>

#include <cute/tensor.hpp>

int main() {
    using namespace cute;

    int storage[12];
    for (int i = 0; i < 12; ++i) {
        storage[i] = i;
    }
    auto matrix = make_tensor(&storage[0], make_layout(make_shape(3, 4), make_stride(4, 1)));

    // _ 保留该 mode；整数固定坐标并去掉该 mode。
    auto row_view = matrix(1, cute::_);
    auto column_view = matrix(cute::_, 2);
    assert(size(row_view) == 4);
    assert(size(column_view) == 3);
    assert(stride<0>(row_view) == 1);
    assert(stride<0>(column_view) == 4);

    std::printf("matrix(1, _):");
    for (int col = 0; col < 4; ++col) {
        assert(row_view(col) == 4 + col);
        std::printf(" %d", row_view(col));
    }
    std::printf("\nmatrix(_, 2):");
    for (int row = 0; row < 3; ++row) {
        assert(column_view(row) == row * 4 + 2);
        std::printf(" %d", column_view(row));
    }
    std::printf("\n");

    column_view(1) = 99;
    assert(row_view(2) == 99);
    assert(storage[6] == 99);
    std::puts("t05 PASS");
}
