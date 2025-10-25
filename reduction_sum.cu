#include <iostream>
#include <chrono>

__global__ void kernel(float* a)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < blockDim.x && iy < blockDim.y)
    {
        extern __shared__ float smem[];
        
        int threadID = ix * blockDim.y + iy;
        smem[1 + threadID] = a[threadID];

        if (!threadID)
            smem[0] = 9;
        __syncthreads();

        while (smem[0] > 1)
        {
            int div = ((int)smem[0])/2;
            
            if (threadID < div)
            {
                smem[1 + threadID] += smem[1 + threadID + div];
            }
            __syncthreads();
            if (!threadID)
            {
                if (((int)smem[0]) % 2)
                    smem[1] += smem[(int)smem[0]];
                smem[0] = div;
            }
            __syncthreads();
        }
        if (!threadID)
            a[0] = smem[1];
    }
}

class Timer
{
public:
    Timer();
    ~Timer();
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end_point;
};

int main()
{
    float a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    float* a_gpu = nullptr;
    cudaMalloc(&a_gpu, sizeof(a));
    cudaMemcpy(a_gpu, a, sizeof(a), cudaMemcpyHostToDevice);

    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(3, 3, 1);

    Timer timer;
    kernel<<<grid_dim, block_dim, sizeof(float) * 10>>>(a_gpu);

    cudaMemcpy(a, a_gpu, sizeof(a), cudaMemcpyDeviceToHost);

    std::cout << "sum is = " << a[0] << std::endl;

    cudaFree(a_gpu);

    return 0;
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
