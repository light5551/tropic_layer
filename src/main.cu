#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 32

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)


// cpu
//    for (size_t i = 0; i < m; i++) {
//        for (size_t j = 0; j < p; j++) {
//            for (size_t k = 0; k < n; k++) {
//                float tmp = mat1[i * n + k] * mat2[k * p + j];
//                out[i * p + j] = max(tmp, out[i * p + j]);
//            }
//        }
//    }

//mat1 m x n
//mat2 n x p
//out m x p
template<typename T>
__global__ void maxPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n, const size_t p) {
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i >= m) || (j >= p))
        return;
    out[i * p + j] = mat1[i * n] + mat2[j];
    for (size_t k{1}; k < n; k++)
        out[i * p + j] = max(out[i * p + j], mat1[i * n + k] + mat2[k * p + j]);
}

//mat1 m x n
//mat2 m x n
//out m x n
template<typename T>
__global__ void maxPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n) {
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i > m) || (j > n))
        return;
    out[i + j] = max(mat1[i + j], mat2[i + j]);
}


//mat1 m x n
//mat2 n x p
//out m x p
template<typename T>
__global__ void minPlusMulKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n, const size_t p) {
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i >= m) || (j >= p))
        return;
    out[i * p + j] = mat1[i * n] + mat2[j];
    for (size_t k{1}; k < n; k++)
        out[i * p + j] = min(out[i * p + j], mat1[i * n + k] + mat2[k * p + j]);
}


//mat1 m x n
//mat2 m x n
//out m x n
template<typename T>
__global__ void minPlusAddKernel(const T *mat1, const T *mat2, T *out,
                                 const size_t m, const size_t n) {
    const auto i{blockIdx.y * blockDim.y + threadIdx.y};
    const auto j{blockIdx.x * blockDim.x + threadIdx.x};
    if ((i > m) || (j > n))
        return;
    out[i + j] = min(mat1[i + j], mat2[i + j]);
}


template<typename T>
void showMtr(const T *vec, size_t size) {
    for (int i = 0; i < size; ++i)
        std::cout << vec[i] << " ";
    std::cout << std::endl;
}

int main() {

    const int m = 3, n = 2, p = 2;

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory

    a = (float *) malloc(sizeof(float) * m * n);
    b = (float *) malloc(sizeof(float) * n * p);
    out = (float *) malloc(sizeof(float) * m * p);

    // Initialize host arrays
    for (int i = 0; i < m * n; i++) {
        a[i] = float(i) + 1;
    }
    for (int i = 0; i < n * p; i++) {
        b[i] = float(i) + 1;
    }
//    for (int i = 0; i < m * p; i++) {
//        out[i] = float(INT_MIN);
//    }

    // Allocate device memory

    CUDA_CHECK(cudaMalloc((void **) &d_a, sizeof(float) * m * n));
    CUDA_CHECK(cudaMalloc((void **) &d_b, sizeof(float) * n * p));
    CUDA_CHECK(cudaMalloc((void **) &d_out, sizeof(float) * m * p));

    // Transfer data from host to device memory
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeof(float) * m * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeof(float) * n * p, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, sizeof(float) * m * p, cudaMemcpyHostToDevice));

    // Executing kernel
    dim3 blocks_per_grid(1);
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
//    minPlusMulKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_out, m, n, p);
    //maxPlusMulKernel<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_out, m, n, p);
    //plusMulKernel<float><<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_out, m, n, p);

    maxPlusMulKernel<float><<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_out,
                                                                    m, n, p);

    //maxPlusAddKernel<float><<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_out,
//                                                                    m, n);

    CUDA_CHECK(cudaMemcpy(out, d_out, sizeof(float) * m * p, cudaMemcpyDeviceToHost));

    showMtr(out, m*n);
    //printf("PASSED\n");
    std::cout << "PASSED" << std::endl;
    // Deallocate device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    // Deallocate host memory
    free(a);
    free(b);
    free(out);
}