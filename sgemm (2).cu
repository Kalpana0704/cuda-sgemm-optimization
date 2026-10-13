// SGEMM optimization ladder: C = A * B (row-major, square N x N, FP32)
//
// Build (Colab / any NVIDIA GPU):
//   nvcc -O3 -arch=native -o sgemm sgemm.cu -lcublas
// Run all kernels:      ./sgemm
// Run one kernel only:  ./sgemm 2        (index from the kernel table below)
//
// Output: a table on stdout + results.csv (kernel,N,ms,gflops,pct_cublas)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// K1: Naive — one thread per output element.
// threadIdx.x maps to the ROW, so the 32 threads of a warp touch 32 different
// rows: reads of A and writes of C are strided by N floats -> uncoalesced.
// ============================================================================
__global__ void sgemm_naive(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
bool launch_naive(int N, const float* A, const float* B, float* C) {
    dim3 block(32, 32), grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
    sgemm_naive<<<grid, block>>>(N, A, B, C);
    return true;
}

// ============================================================================
// K2: Coalesced — same math, but threadIdx.x maps to the COLUMN.
// A warp now reads 32 consecutive floats of B and writes 32 consecutive floats
// of C (one 128-byte transaction), and all threads share the same A element
// (broadcast). Only the index mapping changed.
// ============================================================================
__global__ void sgemm_coalesced(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
bool launch_coalesced(int N, const float* A, const float* B, float* C) {
    dim3 block(32, 32), grid(CEIL_DIV(N, 32), CEIL_DIV(N, 32));
    sgemm_coalesced<<<grid, block>>>(N, A, B, C);
    return true;
}

// ============================================================================
// K3: Shared-memory tiling.
// Each block computes a TILE x TILE (32 x 32) tile of C, one output per thread
// (same thread layout as K2). K is walked in steps of TILE:
//   1. each thread loads ONE element of the A tile and ONE of the B tile into
//      shared memory (coalesced, since tx indexes the column in both loads)
//   2. __syncthreads(): no thread may read the tiles until ALL are loaded
//   3. each thread does TILE multiply-adds reading only from shared memory
//   4. __syncthreads(): no thread may overwrite the tiles (next iteration)
//      while a slower thread is still reading them
// Global loads per output: K2 = 2N, K3 = 2N / TILE  -> 32x fewer.
// Shared-memory reads: As[ty][k] is the same address for the whole warp
// (broadcast); Bs[k][tx] is 32 consecutive floats (32 banks, no conflict).
// ============================================================================
template <int TILE>
__global__ void sgemm_smem(int N, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        // 1. Load one element of each tile (0 if outside the matrix).
        As[ty][tx] = (row < N && t + tx < N) ? A[row * N + (t + tx)] : 0.0f;
        Bs[ty][tx] = (t + ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();                                   // 2.

        #pragma unroll
        for (int k = 0; k < TILE; ++k)                     // 3.
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                                   // 4.
    }
    if (row < N && col < N) C[row * N + col] = acc;
}
bool launch_smem(int N, const float* A, const float* B, float* C) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE), grid(CEIL_DIV(N, TILE), CEIL_DIV(N, TILE));
    sgemm_smem<TILE><<<grid, block>>>(N, A, B, C);
    return true;
}

// ============================================================================
// K4: 2D register blocking (no vectorization yet — that's K5).
// - Each BLOCK computes a 128 x 128 tile of C with 256 threads.
// - Each THREAD computes an 8 x 8 sub-tile (64 outputs) in registers.
// - K is walked in steps of BK = 8 through shared memory.
// Why it's faster than K3: in K3 every multiply-add needs 2 shared-memory
// reads (1 FMA per 2 loads). Here, per k, a thread reads 8 values of A and
// 8 of B into registers and does an 8x8 outer product: 64 FMAs per 16 loads
// (4 FMAs per load) -> 8x better arithmetic intensity at the shared-memory level.
// Cost: ~64 accumulator registers per thread, which limits occupancy.
// Loads here are scalar (4 per thread per tile); K5 replaces them with float4.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_regblock(int N, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);    // 256
    static_assert(NUM_THREADS % BK == 0 && NUM_THREADS % BN == 0, "tile/thread mismatch");

    __shared__ float As[BM][BK];   // 128 x 8 slice of A
    __shared__ float Bs[BK][BN];   // 8 x 128 slice of B

    const int tid = threadIdx.x;
    const int threadRow = tid / (BN / TN);    // 0..15: which 8-row band of the C tile
    const int threadCol = tid % (BN / TN);    // 0..15: which 8-col band of the C tile

    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // Load pattern: 256 threads cover the 128x8 A slice in 4 passes of 32 rows,
    // and the 8x128 B slice in 4 passes of 2 rows.
    const int innerRowA = tid / BK, innerColA = tid % BK;
    constexpr int strideA = NUM_THREADS / BK;    // 32 rows per pass
    const int innerRowB = tid / BN, innerColB = tid % BN;
    constexpr int strideB = NUM_THREADS / BN;    // 2 rows per pass

    float acc[TM][TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += strideA)
            As[innerRowA + off][innerColA] = A[(innerRowA + off) * N + innerColA];
        #pragma unroll
        for (int off = 0; off < BK; off += strideB)
            Bs[innerRowB + off][innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[threadRow * TM + i][k];
            #pragma unroll
            for (int i = 0; i < TN; ++i) regN[i] = Bs[k][threadCol * TN + i];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += regM[m] * regN[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            C[(threadRow * TM + m) * N + threadCol * TN + n] = acc[m][n];
}
bool launch_regblock(int N, const float* A, const float* B, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    if (N % BM != 0) {
        fprintf(stderr, "4_regblock requires N to be a multiple of %d (got %d)\n", BM, N);
        return false;
    }
    dim3 block((BM * BN) / (TM * TN));   // 256 threads
    dim3 grid(N / BN, N / BM);
    sgemm_regblock<BM, BN, BK, TM, TN><<<grid, block>>>(N, A, B, C);
    return true;
}

// ============================================================================
// K5: 2D register blocking + vectorized (float4) loads.
//
// Layout of the work:
//   - Each BLOCK computes a BM x BN = 128 x 128 tile of C.
//   - Each THREAD computes a TM x TN = 8 x 8 sub-tile of that (64 outputs),
//     kept in registers: acc[8][8]. So a block needs 128*128/64 = 256 threads.
//   - K is walked in steps of BK = 8: load a 128x8 slice of A and an 8x128
//     slice of B into shared memory, multiply, repeat.
//
// What is "vectorized" here:
//   1. Global loads: each thread loads ONE float4 (16 bytes) of A and ONE of B
//      per step (128*8 floats / 256 threads = 4 floats each).
//   2. A is stored TRANSPOSED in shared memory (As[k][m]), so the 8 values of
//      A a thread needs for one k are contiguous -> read as two float4.
//   3. B's 8 values per k are already contiguous -> two float4.
//   4. Results are written to C with float4 stores.
//
// Requirements: N % 128 == 0 (tile size) and 16-byte alignment.
// README question: the transposed store into As causes a 2-way bank conflict
// (two threads with the same innerRowA, different innerColA). Why? Is it worth
// fixing, given it happens once per 8 k-steps?
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
sgemm_vec4(int N, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ __align__(16) float As[BK][BM];   // A tile, transposed: As[k][m]
    __shared__ __align__(16) float Bs[BK][BN];   // B tile, normal:     Bs[k][n]

    const int tid = threadIdx.x;

    // Which 8x8 sub-tile of the block's 128x128 C tile this thread owns.
    const int threadRow = tid / (BN / TN);   // 0..15
    const int threadCol = tid % (BN / TN);   // 0..15

    // Move pointers to this block's starting positions.
    A += blockIdx.y * BM * N;                       // first row of A for this block
    B += blockIdx.x * BN;                           // first column of B for this block
    C += blockIdx.y * BM * N + blockIdx.x * BN;     // top-left of this block's C tile

    // Which float4 of the A tile (128 rows x 2 float4) this thread loads.
    const int innerRowA = tid / (BK / 4);    // 0..127
    const int innerColA = tid % (BK / 4);    // 0..1
    // Which float4 of the B tile (8 rows x 32 float4) this thread loads.
    const int innerRowB = tid / (BN / 4);    // 0..7
    const int innerColB = tid % (BN / 4);    // 0..31

    float acc[TM][TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        // ---- 1. Global -> shared, 16 bytes per load ----
        float4 a = reinterpret_cast<const float4*>(&A[innerRowA * N + innerColA * 4])[0];
        As[innerColA * 4 + 0][innerRowA] = a.x;    // transpose while storing
        As[innerColA * 4 + 1][innerRowA] = a.y;
        As[innerColA * 4 + 2][innerRowA] = a.z;
        As[innerColA * 4 + 3][innerRowA] = a.w;

        reinterpret_cast<float4*>(&Bs[innerRowB][innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();          // tiles fully loaded before anyone reads them

        A += BK;                  // next 8 columns of A
        B += BK * N;              // next 8 rows of B

        // ---- 2. Multiply from shared memory into registers ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; i += 4) {
                float4 v = reinterpret_cast<const float4*>(&As[k][threadRow * TM + i])[0];
                regM[i] = v.x; regM[i + 1] = v.y; regM[i + 2] = v.z; regM[i + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < TN; i += 4) {
                float4 v = reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + i])[0];
                regN[i] = v.x; regN[i + 1] = v.y; regN[i + 2] = v.z; regN[i + 3] = v.w;
            }
            // 8x8 outer product: 64 FMAs from 16 shared-memory values.
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += regM[m] * regN[n];
        }

        __syncthreads();          // everyone done reading before the next load overwrites
    }

    // ---- 3. Registers -> global, 16 bytes per store ----
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 out = make_float4(acc[m][n], acc[m][n + 1], acc[m][n + 2], acc[m][n + 3]);
            reinterpret_cast<float4*>(&C[(threadRow * TM + m) * N + threadCol * TN + n])[0] = out;
        }
    }
}
bool launch_vec4(int N, const float* A, const float* B, float* C) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    if (N % BM != 0) {
        fprintf(stderr, "5_vec4 requires N to be a multiple of %d (got %d)\n", BM, N);
        return false;
    }
    dim3 block((BM * BN) / (TM * TN));   // 256 threads
    dim3 grid(N / BN, N / BM);
    sgemm_vec4<BM, BN, BK, TM, TN><<<grid, block>>>(N, A, B, C);
    return true;
}

// K6 (optional): tensor cores with WMMA needs FP16 inputs, so it is not a
// fair FP32 comparison. Add it at the end as a separate experiment.

// ============================================================================
// Harness
// ============================================================================
using LaunchFn = bool (*)(int, const float*, const float*, float*);
struct Kernel { const char* name; LaunchFn launch; };

static const Kernel kKernels[] = {
    {"1_naive",     launch_naive},
    {"2_coalesced", launch_coalesced},
    {"3_smem_tile", launch_smem},
    {"4_regblock",  launch_regblock},
    {"5_vec4",      launch_vec4},
};
static const int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

template <typename F>
float time_ms(F run, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    run();                                   // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) run();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

int main(int argc, char** argv) {
    int only = (argc > 1) ? atoi(argv[1]) : -1;  // 1-based kernel index, -1 = all
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    const int maxN = sizes.back();
    const size_t maxBytes = (size_t)maxN * maxN * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA, maxBytes));
    CUDA_CHECK(cudaMalloc(&dB, maxBytes));
    CUDA_CHECK(cudaMalloc(&dC, maxBytes));
    CUDA_CHECK(cudaMalloc(&dRef, maxBytes));
    std::vector<float> hA((size_t)maxN * maxN), hB(hA.size()), hC(hA.size()), hRef(hA.size());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    FILE* csv = fopen("results.csv", "w");
    fprintf(csv, "kernel,N,ms,gflops,pct_cublas\n");
    printf("%-14s %6s %10s %10s %10s %s\n", "kernel", "N", "ms", "GFLOPS", "%cuBLAS", "");

    srand(42);
    for (int N : sizes) {
        const size_t count = (size_t)N * N, bytes = count * sizeof(float);
        const double flops = 2.0 * N * N * N;
        const int reps = (N >= 4096) ? 3 : 10;

        for (size_t i = 0; i < count; ++i) {
            hA[i] = 2.0f * rand() / RAND_MAX - 1.0f;
            hB[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        }
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

        // cuBLAS is column-major. Row-major C = A*B equals column-major C^T = B^T * A^T,
        // so we pass B first and A second.
        const float alpha = 1.0f, beta = 0.0f;
        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, dB, N, dA, N, &beta, dRef, N));
        };
        float ms_cublas = time_ms(run_cublas, reps);
        double gf_cublas = flops / (ms_cublas * 1e6);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
        printf("%-14s %6d %10.3f %10.1f %9.1f%%\n", "cuBLAS", N, ms_cublas, gf_cublas, 100.0);
        fprintf(csv, "cuBLAS,%d,%.4f,%.2f,100.0\n", N, ms_cublas, gf_cublas);

        for (int k = 0; k < kNumKernels; ++k) {
            if (only > 0 && k + 1 != only) continue;
            const Kernel& ker = kKernels[k];

            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            if (!ker.launch(N, dA, dB, dC)) {
                if (N == sizes.front()) printf("%-14s  (not implemented, skipped)\n", ker.name);
                continue;
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Correctness vs cuBLAS (relative error; FP32 summation order differs).
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
            double max_err = 0.0;
            for (size_t i = 0; i < count; ++i)
                max_err = fmax(max_err, fabs(hC[i] - hRef[i]) / (fabs(hRef[i]) + 1.0));
            if (max_err > 1e-3) {
                printf("%-14s %6d  FAILED correctness (max rel err %.2e) — not timed\n",
                       ker.name, N, max_err);
                continue;
            }

            float ms = time_ms([&]() { ker.launch(N, dA, dB, dC); }, reps);
            double gf = flops / (ms * 1e6);
            printf("%-14s %6d %10.3f %10.1f %9.1f%%\n", ker.name, N, ms, gf, 100.0 * gf / gf_cublas);
            fprintf(csv, "%s,%d,%.4f,%.2f,%.2f\n", ker.name, N, ms, gf, 100.0 * gf / gf_cublas);
        }
        printf("\n");
    }

    fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dRef));
    printf("Wrote results.csv\n");
    return 0;
}
