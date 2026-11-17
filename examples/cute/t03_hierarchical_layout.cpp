#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    // 两个 batch，每个有 6 行 4 列；把行分解为 (2, 3)：row = inner + 2 * outer。
    auto layout = make_layout(make_shape(2, make_shape(2, 3), 4),
                              make_stride(24, make_stride(4, 8), 1));
    std::printf("layout: ");
    print(layout);
    std::printf("\nrank = %d, depth = %d\n", int(rank(layout)), int(depth(layout)));
    std::printf("(batch, (inner, outer), col) -> offset\n");

    assert(rank(layout) == 3);       // 顶层有三个 mode：batch、行和列。
    assert(depth(layout) == 2);      // 行 mode 内还有一层 tuple。
    assert(size<0>(layout) == 2);
    assert(size<1>(layout) == 6);
    assert(size(layout) == 48);

    for (int batch = 0; batch < 2; ++batch) {
        for (int outer = 0; outer < 3; ++outer) {
            for (int inner = 0; inner < 2; ++inner) {
                for (int col = 0; col < 4; ++col) {
                    auto coord = make_coord(batch, make_coord(inner, outer), col);
                    int offset = layout(coord);
                    int row = inner + 2 * outer;
                    assert(offset == (batch * 6 + row) * 4 + col);
                    std::printf("(%d,(%d,%d),%d) -> %2d  ", batch, inner, outer, col, offset);
                }
                std::printf("\n");
            }
        }
    }
    std::puts("t03 PASS");
}
