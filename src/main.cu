#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <vector>

#define TILE_DIM 32
#define BLOCK_DIM 32

#define CUDA_CHECK(err)                                                               \
    do                                                                                \
    {                                                                                 \
        cudaError_t err_ = (err);                                                     \
        if (err_ != cudaSuccess)                                                      \
        {                                                                             \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                        \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// mat1 m x n
// mat2 n x p
// out m x p
template <typename T>
__global__ void maxPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n, const size_t p)
{
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i >= m) || (j >= p))
        return;
    out[i * p + j] = mat1[i * n] + mat2[j];
    for (size_t k{1}; k < n; k++)
        out[i * p + j] = max(out[i * p + j], mat1[i * n + k] + mat2[k * p + j]);
}

// mat1 m x n
// mat2 n x p
// out m x p
template <typename T>
__global__ void optimizedMaxPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                          const size_t m, const size_t n, const size_t p)
{
    __shared__ T mat1Tile[TILE_DIM][TILE_DIM];
    __shared__ T mat2Tile[TILE_DIM][TILE_DIM];
    T val{0};

    for (size_t tile_idx{0};
         tile_idx < ceilf(static_cast<float>(n) / BLOCK_DIM); ++tile_idx)
    {
        size_t i{blockIdx.y * blockDim.y + threadIdx.y};
        size_t j{tile_idx * blockDim.x + threadIdx.x};
        if ((i < m) && (j < n))
        {
            mat1Tile[threadIdx.y][threadIdx.x] = mat1[i * n + j];
        }
        else
        {
            mat1Tile[threadIdx.y][threadIdx.x] = 0;
        }
        i = tile_idx * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if ((i < n) && (j < p))
        {
            mat2Tile[threadIdx.y][threadIdx.x] = mat2[i * p + j];
        }
        else
        {
            mat2Tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (size_t k{0}; k < BLOCK_DIM; ++k)
        {
            val = max(val, mat1Tile[threadIdx.y][k] + mat2Tile[k][threadIdx.x]);
        }
        __syncthreads();
    }

    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    if ((i < m) && (j < p))
    {
        out[i * p + j] = val;
    }
}

// mat1 m x n
// mat2 m x n
// out m x n
template <typename T>
__global__ void maxPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n)
{
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i > m) || (j > n))
        return;
    out[i + j] = max(mat1[i + j], mat2[i + j]);
}

// mat1 m x n
// mat2 m x n
// out m x n
template <typename T>
__global__ void optimizedMaxPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                          const size_t m, const size_t n)
{
}

// mat1 m x n
// mat2 n x p
// out m x p
template <typename T>
__global__ void minPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n, const size_t p)
{
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i >= m) || (j >= p))
        return;
    out[i * p + j] = mat1[i * n] + mat2[j];
    for (size_t k{1}; k < n; k++)
        out[i * p + j] = min(out[i * p + j], mat1[i * n + k] + mat2[k * p + j]);
}

// mat1 m x n
// mat2 n x p
// out m x p
template <typename T>
__global__ void optimizedMinPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                          const size_t m, const size_t n, const size_t p)
{
}

// mat1 m x n
// mat2 m x n
// out m x n
template <typename T>
__global__ void minPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n)
{
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i > m) || (j > n))
        return;
    out[i + j] = min(mat1[i + j], mat2[i + j]);
}

// mat1 m x n
// mat2 m x n
// out m x n
template <typename T>
__global__ void optimizedMinPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                          const size_t m, const size_t n)
{
}

template <typename T>
void showMtr(const T *vec, size_t size)
{
    for (int i = 0; i < size; ++i)
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}

template <typename T>
void mm_cuda(T const *mat_1, T const *mat_2, T *mat_3, size_t m, size_t n,
             size_t p,
             void (*fun)(T const *, T const *, T *, size_t, size_t, size_t))
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    fun<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n, p);
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Tropic Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
__global__ void mm_kernel_optimized(T const *mat_1, T const *mat_2, T *mat_3,
                                    size_t m, size_t n, size_t p)
{
    __shared__ T mat_1_tile[BLOCK_DIM][BLOCK_DIM];
    __shared__ T mat_2_tile[BLOCK_DIM][BLOCK_DIM];

    T acc_sum{0};

    for (size_t tile_idx{0};
         tile_idx < ceilf(static_cast<float>(n) / BLOCK_DIM); ++tile_idx)
    {
        size_t i{blockIdx.y * blockDim.y + threadIdx.y};
        size_t j{tile_idx * blockDim.x + threadIdx.x};
        if ((i < m) && (j < n))
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = mat_1[i * n + j];
        }
        else
        {
            mat_1_tile[threadIdx.y][threadIdx.x] = 0;
        }
        i = tile_idx * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if ((i < n) && (j < p))
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = mat_2[i * p + j];
        }
        else
        {
            mat_2_tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (size_t k{0}; k < BLOCK_DIM; ++k)
        {
            acc_sum += mat_1_tile[threadIdx.y][k] * mat_2_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    if ((i < m) && (j < p))
    {
        mat_3[i * p + j] = acc_sum;
    }
}

template <typename T>
__global__ void copy(T *odata, const T *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += 8)
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

template <typename T>
__global__ void mm_kernel(T const *mat_1, T const *mat_2, T *mat_3, size_t m,
                          size_t n, size_t p)
{
    size_t i{blockIdx.y * blockDim.y + threadIdx.y};
    size_t j{blockIdx.x * blockDim.x + threadIdx.x};

    if ((i >= m) || (j >= p))
    {
        return;
    }

    T acc_sum{0};
    for (size_t k{0}; k < n; ++k)
    {
        acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
    }
    mat_3[i * p + j] = acc_sum;
}

template <typename T>
void bestCopyTime(T const *mat_1, T const *mat_2, float &time)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_DIM)
        mat_1[(y + j) * width + x] = mat_2[(y + j) * width + x];

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
}

int main()
{

    const int m = 1000, n = 1000, p = 1000;

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;
    cudaEvent_t start, stop;
    float gpu_elapsed_time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host memory

    a = (float *)malloc(sizeof(float) * m * n);
    b = (float *)malloc(sizeof(float) * n * p);
    out = (float *)malloc(sizeof(float) * m * p);

    // Initialize host arrays
    for (int i = 0; i < m * n; i++)
    {
        a[i] = float(i) + 1;
    }
    for (int i = 0; i < n * p; i++)
    {
        b[i] = float(i) + 1;
    }

    // Allocate device memory

    CUDA_CHECK(cudaMalloc((void **)&d_a, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc((void **)&d_b, sizeof(float) * n * p));
    CUDA_CHECK(cudaMalloc((void **)&d_out, sizeof(float) * m * p));

    // Transfer data from host to device memory
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * n * p, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, sizeof(float) * m * p, cudaMemcpyHostToDevice));

    // Executing kernel
    mm_cuda(d_a, d_b, d_out, m, n, p, optimizedMaxPlusMulKernel);

    cudaEventRecord(start, 0);
    mm_cuda(d_a, d_b, d_out, m, n, p, optimizedMaxPlusMulKernel);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    std::cout << "Time: " << gpu_elapsed_time_ms << std::endl;

    cudaEventRecord(start, 0);
    mm_cuda(d_a, d_b, d_out, m, n, p, mm_kernel);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    std::cout << "Time: " << gpu_elapsed_time_ms << std::endl;

    cudaEventRecord(start, 0);
    mm_cuda(d_a, d_b, d_out, m, n, p, mm_kernel_optimized);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    std::cout << "Time: " << gpu_elapsed_time_ms << std::endl;

    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));

    cudaEventRecord(start, 0);
    copy<<<blocks_per_grid, threads_per_block>>>(d_a, d_b);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    std::cout << "Time: " << gpu_elapsed_time_ms << std::endl;

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * m * p, cudaMemcpyDeviceToHost));

    // showMtr(out, m * n);
    std::cout << "PASSED" << std::endl;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    free(a);
    free(b);
    free(out);
}
