#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    // 把 6 行分解为 (2, 3)：row = inner + 2 * outer。
    auto layout = make_layout(make_shape(make_shape(2, 3), 4),
                              make_stride(make_stride(4, 8), 1));
    std::printf("layout: ");
    print(layout);
    std::printf("\n((inner, outer), col) -> offset\n");

    assert(rank(layout) == 2);       // 顶层有两个 mode：行和列。
    assert(depth(layout) == 2);      // 行 mode 内还有一层 tuple。
    assert(size<0>(layout) == 6);
    assert(size(layout) == 24);

    for (int outer = 0; outer < 3; ++outer) {
        for (int inner = 0; inner < 2; ++inner) {
            for (int col = 0; col < 4; ++col) {
                auto coord = make_coord(make_coord(inner, outer), col);
                int offset = layout(coord);
                int row = inner + 2 * outer;
                assert(offset == row * 4 + col);
                std::printf("((%d,%d),%d) -> %2d  ", inner, outer, col, offset);
            }
            std::printf("\n");
        }
    }
    std::puts("t03 PASS");
}
