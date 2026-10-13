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
// K3: Shared-memory tiling — YOUR TURN.
// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
//   1. each thread loads ONE element of the A tile and ONE of the B tile
//      into __shared__ float As[TILE][TILE], Bs[TILE][TILE]
//   2. __syncthreads()   (why is this needed?)
//   3. accumulate TILE products reading only from shared memory
//   4. __syncthreads()   (why a second one?)
// README question: how many global loads per output element vs K2?
// ============================================================================
template <int TILE>
__global__ void sgemm_smem(int N, const float* A, const float* B, float* C) {
    // TODO
}
bool launch_smem(int N, const float* A, const float* B, float* C) {
    // TODO: launch sgemm_smem<32> with block(32, 32), then return true.
    return false;  // false = "not implemented yet", harness skips it
}

// ============================================================================
// K4: Register blocking — YOUR TURN.
// Each thread computes several outputs (start with TM = 8 outputs in one
// column = "1D blocktiling", then TM x TN = 8 x 8 = "2D blocktiling").
// Keep partial sums in a register array: float acc[TM][TN].
// Fewer threads per block, much more work per thread.
// README question: arithmetic intensity (FLOPs per shared-memory load)
// before vs after. Why does that matter?
// ============================================================================
__global__ void sgemm_regblock(int N, const float* A, const float* B, float* C) {
    // TODO
}
bool launch_regblock(int N, const float* A, const float* B, float* C) {
    return false;
}

// ============================================================================
// K5: Vectorized loads — YOUR TURN.
// Load global memory with float4 (reinterpret_cast<const float4*>) so each
// instruction moves 16 bytes. Common trick: store the A tile TRANSPOSED in
// shared memory so the inner loop can also read it with float4.
// Requires N % 4 == 0 and 16-byte-aligned pointers (cudaMalloc guarantees it).
// README question: did you introduce shared-memory bank conflicts? How did you
// check?
// ============================================================================
__global__ void sgemm_vec4(int N, const float* A, const float* B, float* C) {
    // TODO
}
bool launch_vec4(int N, const float* A, const float* B, float* C) {
    return false;
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
