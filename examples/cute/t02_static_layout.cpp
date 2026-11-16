#include <cassert>
#include <cstdio>

#include <cute/layout.hpp>

int main() {
    using namespace cute;

    auto dynamic_layout = make_layout(make_shape(3, 4), make_stride(4, 1));
    constexpr auto static_layout = make_layout(
        make_shape(Int<3>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));

    // Int<N> 把 N 编进类型；普通 int 则把数值保存在对象中。
    static_assert(decltype(size(static_layout))::value == 12);
    static_assert(static_layout(2, 3) == 11);

    std::printf("dynamic: ");
    print(dynamic_layout);
    std::printf("\nstatic:  ");
    print(static_layout);
    std::printf("\n");

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            assert(dynamic_layout(row, col) == static_layout(row, col));
        }
    }

    // shape 和 stride 可以混用 static / dynamic 整数。
    int rows = 3;
    auto mixed_layout = make_layout(make_shape(rows, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    assert(mixed_layout(2, 3) == 11);
    std::puts("t02 PASS");
}
