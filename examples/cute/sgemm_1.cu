/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Kernel and launch configuration from NVIDIA CUTLASS v4.8.1.
// https://github.com/NVIDIA/cutlass/blob/v4.8.1/examples/cute/tutorial/sgemm_1.cu

#include <cstdio>
#include <cstdlib>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/util/helper_cuda.hpp>

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK,// (256, 384, 24)
            CtaTiler cta_tiler,    // (128, 128, 8)
            TA const* A,
            AStride dA,            // (1, 256)
            ASmemLayout sA_layout, // (128, 8), stride (1, 128)
            AThreadLayout tA,      // (32, 8), stride (1, 32)
            TB const* B,
            BStride dB,            // (1, 384)
            BSmemLayout sB_layout, // (128, 8), stride (1, 128)
            BThreadLayout tB,      // (32, 8), stride (1, 32)
            TC* C,
            CStride dC,            // (1, 256)
            CSmemLayout,           // (128, 128), stride (1, 128)
            CThreadLayout tC,      // (16, 16), stride (1, 16)
            Alpha alpha,
            Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K) = (256, 384, 24), rank = 3
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K) = (128, 128, 8), rank = 3

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));                          // NumThreads = 256
  CUTE_STATIC_ASSERT_V(size(tC) == size(tA));                          // NumThreads = 256

  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == Int<0>{});  // BLK_M / THR_M = 128 / 32 = 4
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == Int<0>{});  // BLK_K / THR_K = 8 / 8 = 1
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == Int<0>{});  // BLK_N / THR_N = 128 / 32 = 4
  CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == Int<0>{});  // BLK_K / THR_K = 8 / 8 = 1
  CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == Int<0>{});  // BLK_M / THR_M = 128 / 16 = 8
  CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == Int<0>{});  // BLK_N / THR_N = 128 / 16 = 8

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M = 128
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M = 128
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N = 128
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N = 128
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K = 8
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K = 8

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA)); // dA strides for shape MK = (1, 256) for (256, 24)
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB)); // dB strides for shape NK = (1, 384) for (384, 24)
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC)); // dC strides for shape MN = (1, 256) for (256, 384)

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K) = (256, 24)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K) = (384, 24)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N) = (256, 384)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k) = (0..1, 0..2, _)
  /* md
   * `mA` 的 shape 是 `(256, 24)`，按 `(128, 8)` 分块后有 2 个 M tile 和 3 个 K tile。
   *
   * - `make_coord(blockIdx.x, _)` 固定 M tile，保留所有 K tile，得到 `(128, 8, 3)`。
   * - `make_coord(_, _)` 保留所有 M 和 K tile，得到 `(128, 8, 2, 3)`，
   *   坐标顺序是 `(m_in, k_in, m_tile, k_tile)`。
   *
   * 对后一个 view 用 `(_, _, blockIdx.x, _)` 切片，就得到下面的 `gA`。
   */
  Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));  // (BLK_M,BLK_K,k) = (128, 8, 3)
  Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));  // (BLK_N,BLK_K,k) = (128, 8, 3)
  Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));  // (BLK_M,BLK_N)   = (128, 128)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K) = (128, 8)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K) = (128, 8)

  // Partition the copying of A and B tiles across the threads
  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

  // local_tile 算 block 对应哪个 tile， local_partition 算 thread 对应 tile 里哪些 float。
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);   // (THR_M,THR_K,k) = (4, 1, 3)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);   // (THR_M,THR_K)   = (4, 1)
  Tensor tBgB = local_partition(gB, tB, threadIdx.x);   // (THR_N,THR_K,k) = (4, 1, 3)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);   // (THR_N,THR_K)   = (4, 1)
  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA)); // THR_M = 4
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // THR_K = 1
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB)); // THR_N = 4
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // THR_K = 1

  // Define A/B partitioning and C accumulators
  // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

  // Partition sA (BLK_M, BLK_K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K) = (8, 8)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K) = (8, 8)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N) = (8, 8)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N) = (8, 8)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M = 8
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M = 8
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N = 8
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N = 8
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K = 8

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
  //           and then computes on those tiles.
  //   copy(.) operates on the global and shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared and register memory via the tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
    //   Tensor tAgAk = tAgA(_,_,k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    //       }
    //     }
    //   }

    __syncthreads();         // Wait for all threads to read from smem
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);

  // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent to
  //   CUTE_UNROLL
  //   for (int i = 0; i < size(tCrC); ++i) {
  //     tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  //   }
}

// Setup params for an NT GEMM
// Use m-major smem sA, n-major smem sB, and mn-major threads tA|tB
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m); // 256
  auto N = int(n); // 384
  auto K = int(k); // 24
  auto prob_shape = make_shape(M, N, K); // (M, N, K) = (256, 384, 24)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K) = (128, 128, 8)

  // Define the smem layouts (static)
  // make_layout is col-major by default, so sA is m-major and sB is n-major
  auto sA = make_layout(make_shape(bM, bK)); // (m,k) -> smem_idx; m-major = (128, 8)
  auto sB = make_layout(make_shape(bN, bK)); // (n,k) -> smem_idx; n-major = (128, 8)
  auto sC = make_layout(make_shape(bM, bN)); // (m,n) -> smem_idx; m-major = (128, 128)

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

  dim3 dimBlock(size(tC)); // size(tC) = 16*16 = 256 threads per block
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN))); // vx=256/128=2, vy=384/128=3, so dimGrid=(2,3)
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

// Setup params for a TN GEMM
// Use k-major smem sA and sB, and k-major threads tA|tB
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major

  dim3 dimBlock(size(tC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

namespace {

template <bool is_nt>
void check_gemm()
{
  // 对齐 (128, 128, 8)，覆盖多个 CTA 和多个 K tile。
  constexpr int m = 256;
  constexpr int n = 384;
  constexpr int k = 24;
  constexpr float alpha = 2;
  constexpr float beta = 3;
  constexpr float initial_c = -1;

  thrust::host_vector<float> h_A(m * k);
  thrust::host_vector<float> h_B(n * k);
  thrust::host_vector<float> h_C(m * n, initial_c);
  // 小整数的乘加可用 FP32 精确表示，方便逐元素比较。
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> values(-4, 4);
  for (float& value : h_A) value = values(rng);
  for (float& value : h_B) value = values(rng);

  thrust::device_vector<float> d_A = h_A;
  thrust::device_vector<float> d_B = h_B;
  thrust::device_vector<float> d_C = h_C;
  if constexpr (is_nt) {
    gemm_nt(m, n, k, alpha, d_A.data().get(), m, d_B.data().get(), n,
            beta, d_C.data().get(), m);
  } else {
    gemm_tn(m, n, k, alpha, d_A.data().get(), k, d_B.data().get(), k,
            beta, d_C.data().get(), m);
  }
  CUTE_CHECK_LAST();
  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
  h_C = d_C;

  const char* mode = is_nt ? "NT" : "TN";
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      float dot = 0;
      for (int p = 0; p < k; ++p) {
        const int a_offset = is_nt ? row + p * m : row * k + p;
        const int b_offset = is_nt ? col + p * n : col * k + p;
        dot += h_A[a_offset] * h_B[b_offset];
      }

      const float expected = alpha * dot + beta * initial_c;
      const float actual = h_C[row + col * m];
      if (actual != expected) {
        std::fprintf(stderr, "sgemm_1 %s C(%d,%d): expected %g, got %g\n",
                     mode, row, col, expected, actual);
        std::exit(EXIT_FAILURE);
      }
    }
  }
  std::printf("sgemm_1 %s PASS (M=%d N=%d K=%d)\n", mode, m, n, k);
}

}  // namespace

int main()
{
  CUTE_CHECK_ERROR(cudaSetDevice(0));
  check_gemm<true>();
  // check_gemm<false>();
}
