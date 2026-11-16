#include <cassert>
#include <cstdio>

#include <cute/tensor.hpp>

int main() {
    using namespace cute;

    int storage[12];
    for (int i = 0; i < 12; ++i) {
        storage[i] = i;
    }

    // 这个 Tensor 是 storage 的 view；storage 必须比 view 活得更久。
    auto matrix = make_tensor(&storage[0], make_layout(make_shape(3, 4), make_stride(4, 1)));
    std::printf("matrix layout: ");
    print(matrix.layout());
    std::printf("\n");
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            assert(matrix(row, col) == storage[row * 4 + col]);
            std::printf("%3d", matrix(row, col));
        }
        std::printf("\n");
    }

    // 交换 shape 和 stride 得到 transpose view，没有搬运数据。
    auto transposed = make_tensor(&storage[0], make_layout(make_shape(4, 3), make_stride(1, 4)));
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            assert(transposed(col, row) == matrix(row, col));
        }
    }
    transposed(2, 1) = 99;
    assert(matrix(1, 2) == 99);
    assert(storage[6] == 99);
    std::printf("transposed(2,1) = matrix(1,2) = storage[6] = %d\n", storage[6]);
    std::puts("t04 PASS");
}
