#include <cassert>
#include <cstdio>

#include <cute/tensor.hpp>

int main() {
    using namespace cute;

    int storage[48];
    for (int i = 0; i < 48; ++i) {
        storage[i] = i;
    }
    auto matrix = make_tensor(&storage[0], make_layout(make_shape(4, 12), make_stride(12, 1)));
    auto tile_shape = make_shape(Int<2>{}, Int<4>{});
    assert(size<0>(matrix) % size<0>(tile_shape) == 0);
    assert(size<1>(matrix) % size<1>(tile_shape) == 0);

    /* md
     * matrix shape 为 `(4, 12)`，tile shape 为 `(2, 4)`：
     * - 行方向有 `4 / 2 = 2` 个 tile。
     * - 列方向有 `12 / 4 = 3` 个 tile。
     * - tile grid 为 `(2, 3)`，共 6 个 tile。
     *
     * tile `(r, c)` 的起点是 matrix `(r * 2, c * 4)`。
     */
    for (int tile_row = 0; tile_row < 2; ++tile_row) {
        for (int tile_col = 0; tile_col < 3; ++tile_col) {
            // tile_coord 的单位是 tile；(1, 1) 的起点是 matrix(2, 4)。
            auto tile = local_tile(matrix, tile_shape, make_coord(tile_row, tile_col));
            assert(size<0>(tile) == 2);
            assert(size<1>(tile) == 4);
            assert(stride<0>(tile) == 12);  // view 仍沿用原矩阵的行间距。
            std::printf("tile(%d,%d):\n", tile_row, tile_col);
            for (int row = 0; row < 2; ++row) {
                for (int col = 0; col < 4; ++col) {
                    int global_row = tile_row * 2 + row;
                    int global_col = tile_col * 4 + col;
                    assert(tile(row, col) == global_row * 12 + global_col);
                    std::printf("%3d", tile(row, col));
                }
                std::printf("\n");
            }
        }
    }
    std::puts("t06 PASS");
}
