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

void strassens(float* a, float* b, float* c, size_t M, size_t N, size_t K, size_t Mi, size_t Ni, size_t Ki, size_t off_m, size_t off_n, size_t off_k)
{

    if (Mi > 2 || Ni > 2 || Ki > 2)
    {
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + 0,    off_n + 0,      off_k + 0);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + 0,    off_n + Ni/2,   off_k + 0);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + 0,    off_n + 0,      off_k + Ki/2);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + 0,    off_n + Ni/2,   off_k + Ki/2);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + Mi/2, off_n + 0,      off_k + 0);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + Mi/2, off_n + Ni/2,   off_k + 0);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + Mi/2, off_n + 0,      off_k + Ki/2);
        strassens(a, b, c, M, N, K, Mi/2, Ni/2, Ki/2, off_m + Mi/2, off_n + Ni/2,   off_k + Ki/2);
    }

    else if (Mi == 2 && Ni == 2 && Ki == 2)
    {
        float a11 = a[off_m * N + off_n];
        float a12 = a[off_m * N + off_n + 1];
        float a21 = a[off_m * N + off_n + N];
        float a22 = a[off_m * N + off_n + N + 1];

        float b11 = b[off_n * K + off_k];
        float b12 = b[off_n * K + off_k + 1];
        float b21 = b[off_n * K + off_k + N];
        float b22 = b[off_n * K + off_k + K + 1];

        float p1 = a11 * (b12 - b22);
        float p2 = (a11 + a12) * b22;
        float p3 = (a21 + a22) * b11;
        float p4 = a22 * (b21 - b11);
        float p5 = (a11 + a22) * (b11 + b22);
        float p6 = (a12 - a22) * (b21 + b22);
        float p7 = (a11 - a21) * (b11 + b12);

        c[off_m * K + off_k]            += p5 + p4 - p2 + p6;
        c[off_m * K + off_k + 1]        += p1 + p2;
        c[off_m * K + off_k + K ]       += p3 + p4;
        c[off_m * K + off_k + K + 1]    += p1 + p5 - p3 - p7;
        
    }
    
    else
    {
        // 1
    }
}

int main()
{
    size_t M = 256, N = 256, K = 256;

    float a[256*256] = {
        1, 6, 3, 4,
        4, 5, 6, 4,
        4, 6, 3, 4,
        4, 9, 6, 4,
    };

    float b[256*256] = {
        9, 6, 2, 4,
        2, 5, 6, 9,
        1, 9, 3, 4,
        4, 5, 9, 4,
    };

    float c[256*256];
    memset(c, 0, 256*256 * sizeof(float)); // +=

    Timer timer;

#ifdef NAIVE
    matmul(a, b, c, M, N, K);
#else
    strassens(a, b, c, M, N, K, M, N, K, 0, 0, 0);
#endif

    // print(c, M, K);

    return 0;
}