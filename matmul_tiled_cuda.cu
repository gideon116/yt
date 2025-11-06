#include <iostream>
#include <chrono>
#include <random>

constexpr int TILE = 32; // while blockdim = 32*32

class Timer
{
public:
    Timer();
    ~Timer();
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end_point;
};

void cpu(float* m1, float* m2, float* m3, const int M, const int N, const int K);

__global__ void kernel_naive(float* m1, float* m2, float* m3, const int M, const int N, const int K)
{

    int m_i = blockIdx.x * TILE + (threadIdx.x / TILE);
    int k_i = blockIdx.y * TILE + (threadIdx.x % TILE);

    if (m_i < M && k_i < K)
    {
        
        float temp = 0.0f;
        for (int n_i = 0; n_i < N; n_i++)
        {
            temp += m1[m_i * N + n_i] * m2[n_i * K + k_i];
        }

        m3[m_i * K + k_i] = temp;
    }
}


__global__ void kernel_tiled(const float* a, const float* b, float* c, const int M, const int N, const int K)
{
    // int block_level_thread_id = threadIdx.x * blockDim.y + threadIdx.y;
    // int threads_per_block = blockDim.x * blockDim.y;

    // int block_id = blockIdx.x * gridDim.y + blockIdx.y;
    // int global_thread_id = block_id * threads_per_block + block_level_thread_id;
    
    int div = threadIdx.x / TILE;
    int mod = threadIdx.x % TILE;
    int div_TILE = div * TILE;

    int m_i = blockIdx.x * TILE + div;
    int k_i = blockIdx.y * TILE + mod;

    extern __shared__ float s[];
    float* s_a = s;
    float* s_b = s + TILE * TILE;

    float temp = 0.0f;

    for (int t_i = 0; t_i < N; t_i += TILE)
    {
        int n_i_a = t_i + mod;
        int n_i_b = t_i + div;
        
        if (n_i_a < N && n_i_b < N)
        {
            s_a[div_TILE + mod] = a[m_i * N + n_i_a];
            s_b[div_TILE + mod] = b[n_i_b * K + k_i];
        }

        __syncthreads();

        for (int i = 0; i < min(TILE, N - t_i); i++) 
            temp += s_a[div_TILE + i] * s_b[i * TILE + mod];

        __syncthreads();

    }
    if (m_i < M && k_i < K)
        c[m_i * K + k_i] = temp;

}

int main()
{
    constexpr int M = 4, N = 1024, K = 4; 
    std::mt19937 m_g;
    std::normal_distribution<float> m_dist = std::normal_distribution<float>(0.0f, 5.0f);

    float* a = new float[M * N];
    float* b = new float[N * K];
    float* c = new float[M * K];
    float* c_cpu = new float[M * K];

    for (size_t i = 0; i < M * N; i++)
        a[i] = m_dist(m_g);
    for (size_t i = 0; i < N * K; i++)
        b[i] = m_dist(m_g);
    for (size_t i = 0; i < M * K; i++)
    {
        c_cpu[i] = m_dist(m_g);
        c[i] = c_cpu[i];
    }

    cpu(a, b, c_cpu, M, N, K);


    float *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, sizeof(float) * M * N);
    cudaMalloc(&b_gpu, sizeof(float) * N * K);
    cudaMalloc(&c_gpu, sizeof(float) * M * K);

    cudaMemcpy(a_gpu, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    auto ceil_div = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };

    dim3 grid(ceil_div(M, TILE), ceil_div(K, TILE));
    dim3 block(TILE*TILE);

    Timer timer;
    kernel_tiled<<<grid, block, 2 * TILE * TILE * sizeof(float)>>>(a_gpu, b_gpu, c_gpu, M, N, K);

    cudaMemcpy(c, c_gpu, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * K; i++)
    {
        if (c[i] - c_cpu[i] > 0.1 || c_cpu[i] - c[i] > 0.1)
        {
            std::cout << "error: " << c[i] << " - " << c_cpu[i] << " = " << c[i] - c_cpu[i] << std::endl;
            break;
        }
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_cpu;
}


void cpu(float* m1, float* m2, float* m3, const int M, const int N, const int K)
{
    for (int m_i = 0; m_i < M; m_i++)
    {
        for (int k_i = 0; k_i < K; k_i++)
        {
            float temp = 0;
            for (int n_i = 0; n_i < N; n_i++)
            {
                temp += m1[m_i * N + n_i] * m2[n_i * K + k_i];
            }
            m3[m_i * K + k_i] = temp;
        }
    }
}


Timer::Timer() { m_start_point = std::chrono::high_resolution_clock::now(); }
Timer::~Timer()
{
    m_end_point = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_start_point);
    auto end = std::chrono::time_point_cast<std::chrono::microseconds>(m_end_point);
    auto duration = end - start;
    float sec = duration.count() * 0.000001f;
    std::cout << sec << " sec" << "\n";
}


