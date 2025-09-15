/*
By Gideon Kassa
Sep 14, 2025

Wef
Dartmouth College
*/

#include <iostream>
#include <cstring>
#include <chrono>

class Timer
{
public:
    Timer() { m_start_point = std::chrono::high_resolution_clock::now(); }
    ~Timer()
    {
        m_end_point = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_start_point);
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(m_end_point);
        auto duration = end - start;
        float sec = duration.count() * 0.000001f;
        std::cout << sec << " sec" << "\n";
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end_point;
};

void matmul(float* a, float* b, float* c, size_t M, size_t N, size_t K)
{
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t mi = 0; mi < M; mi++)
    {
        for (size_t ni = 0; ni < N; ni++)
        {
            for (size_t ki = 0; ki < K; ki++)
            {
                c[mi*K + ki] += a[mi*N + ni] * b[ni*K + ki];
            }
        }
    }
}

void print(float* matrix, size_t R, size_t C)
{
    for (size_t ri = 0; ri < R; ri++)
    {
        for (size_t ci = 0; ci < C; ci++)
            std::cout << matrix[ri * C + ci] << ", ";
        std::cout << '\n';
    }
    std::cout << "______________\n" ;
}

struct PC
{
    int M, N, K;
};

__global__ void k_mul(float* a_gpu, float* b_gpu, float* c_gpu, PC pc)
{
    uint ri = blockIdx.x * blockDim.x + threadIdx.x;
    uint ci = blockIdx.y * blockDim.y + threadIdx.y;

    if (ri < pc.M && ci < pc.K)
    {
        uint a_gpu_i = ri * pc.N;
        uint b_gpu_i = ci;

        float sum = 0.0;
        for (uint j = 0; j < pc.N; j++)
        {
            sum += a_gpu[a_gpu_i + j] * b_gpu[j * pc.K + b_gpu_i];
        }

        uint c_gpu_i = ri * pc.K + ci;
        c_gpu[c_gpu_i] = sum;
    }

}

int main()
{
    int M = 1064, N = 1064, K = 1064;

    float a[1064*1064] = {
        1, 6, 3, 4,
        4, 5, 6, 4,
        4, 6, 3, 4,
        4, 9, 6, 4,
    };

    float b[1064*1064] = {
        9, 6, 2, 4,
        2, 5, 6, 9,
        1, 9, 3, 4,
        4, 5, 9, 4,
    };

    float c[1064*1064];
    memset(c, 0, 1064*1064 * sizeof(float));

#if 0
    Timer timer;
    matmul(a, b, c, M, N, K);
    // print(c, M, K);

    memset(c, 0, sizeof(float) * M * K);
    // print(c, M, K);

#else
    
    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* c_gpu = nullptr;

    cudaMalloc(&a_gpu, sizeof(float) * M * N);
    cudaMalloc(&b_gpu, sizeof(float) * N * K);
    cudaMalloc(&c_gpu, sizeof(float) * M * K);

    cudaMemcpy(a_gpu, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    /// operation matmul
    auto cil_div = [](int x, int y){return (x + y - 1) / y ; };
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(cil_div(M, dimBlock.x), cil_div(N, dimBlock.y), 1);

    PC pc = {M, N, K};
    Timer timer;
    k_mul<<<dimBlock, dimGrid>>>(a_gpu, b_gpu, c_gpu, pc);
    cudaMemcpy(c, c_gpu, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    // print(c, M, K);
#endif 



    return 0;
}

