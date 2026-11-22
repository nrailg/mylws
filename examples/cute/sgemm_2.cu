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
// https://github.com/NVIDIA/cutlass/blob/v4.8.1/examples/cute/tutorial/sgemm_2.cu

#include <cstdio>
#include <cstdlib>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/util/helper_cuda.hpp>

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK,// (256, 384, 24)
            CtaTiler cta_tiler,    // (128, 128, 8)
            TA const* A,
            AStride dA,            // (1, 256)
            ASmemLayout sA_layout, // (128, 8), stride (1, 128)
            TiledCopyA copy_a,     // (32, 8) threads, (4, 1) values per thread
            TB const* B,
            BStride dB,            // (1, 384)
            BSmemLayout sB_layout, // (128, 8), stride (1, 128)
            TiledCopyB copy_b,     // (32, 8) threads, (4, 1) values per thread
            TC* C,
            CStride dC,            // (1, 256)
            CSmemLayout,           // (128, 128), stride (1, 128)
            TiledMma mma,          // (16, 16, 1) threads, one scalar FMA per atom
            Alpha alpha,
            Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K) = (256, 384, 24), rank = 3
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K) = (128, 128, 8), rank = 3

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads = 256
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads = 256

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
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Prefetch the first K tile from global memory into registers.
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);

  // Define A/B partitioning and C accumulators
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M = 8
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N = 8
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K = 8

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

  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    __syncthreads();         // Wait for all threads to finish reading the previous tile.
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Make the current tile visible to all threads.

    // Prefetch into registers before computing on the current shared memory tile.
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);

    gemm(mma, tCsA, tCsB, tCrC);
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

  // Each of the 32x8 threads copies four contiguous floats per 128-bit load.
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{},
									Layout<Shape<_4,_1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{},
									Layout<Shape<_4,_1>>{});
  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{}, Layout<Shape<_16,_16,_1>>{});

  dim3 dimBlock(size(mmaC)); // 16*16 = 256 threads per block
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN))); // vx=256/128=2, vy=384/128=3, so dimGrid=(2,3)
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
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
	assert(false);
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
        std::fprintf(stderr, "sgemm_2 %s C(%d,%d): expected %g, got %g\n",
                     mode, row, col, expected, actual);
        std::exit(EXIT_FAILURE);
      }
    }
  }
  std::printf("sgemm_2 %s PASS (M=%d N=%d K=%d)\n", mode, m, n, k);
}

}  // namespace

int main()
{
  CUTE_CHECK_ERROR(cudaSetDevice(0));
  check_gemm<true>();
}
