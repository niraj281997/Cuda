#include<iostream>
#include<cuda_runtime.h>
#include<cstdlib>
#include<cstdio>

using namespace std;

// --- CUDA Parameters ---
#define TILE 32 // Block dimension size
#define CHECK_CUDA_ERROR(val) check(val, #val, __FILE__ , __LINE__)

// --- Error Checking Function ---
template <class T>
void check(T err , const char * func, const char * file , const int line)
{
    if(err != cudaSuccess)
    {
        // In a proper application, this should report the error string
        cerr << "CUDA Error encountered: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// --- Utility Functions (Provided by User) ---
void fill_random(int *arr, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        arr[i] = rand() % 10;
    }
}

void print_matrix(int *arr, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%4d", arr[r * cols + c]);
        }
        printf("\n");
    }
}

// =========================================================================
// 1. Tiled Matrix Multiplication Kernel (SGEMM)
// =========================================================================

__global__ void f_mat_mul_tiled(int *A, int *B, int *C, const int M, const int N, const int K)
{
    // Shared memory for tiles of A and B
    __shared__ int sA[TILE][TILE];
    __shared__ int sB[TILE][TILE];

    // Global coordinates for the C element this thread computes
    int row = blockIdx.y * TILE + threadIdx.y; // Row index of C
    int col = blockIdx.x * TILE + threadIdx.x; // Column index of C

    int sum = 0;

    // Loop over the tiles required to compute C[row][col]
    // The number of tiles along the K dimension (inner product) is ceil(K / TILE)
    int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++)
    {
        // 1. Load Phase: Each thread loads one element from Global Memory to Shared Memory

        // Coordinates for the elements to load into the current tile:
        // A element: A[row][t*TILE + threadIdx.x]
        // B element: B[t*TILE + threadIdx.y][col]
        
        int row_A = row;
        int col_A = t * TILE + threadIdx.x;

        int row_B = t * TILE + threadIdx.y;
        int col_B = col;

        // Perform boundary checks for matrices A and B
        if (row_A < M && col_A < K) {
            sA[threadIdx.y][threadIdx.x] = A[row_A * K + col_A];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0; // Pad with zero if out of bounds
        }

        if (row_B < K && col_B < N) {
            sB[threadIdx.y][threadIdx.x] = B[row_B * N + col_B];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0; // Pad with zero if out of bounds
        }

        // Wait for all threads in the block to finish loading both tiles
        __syncthreads();

        // 2. Compute Phase: Inner loop uses fast Shared Memory access
        for (int k = 0; k < TILE; k++)
        {
            // C[row][col] += A[row][k] * B[k][col]
            // The thread accesses its loaded tiles sA[row_offset][k] and sB[k][col_offset]
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Wait for all threads in the block to finish computation for the current tile
        // before the next tile is loaded (prevents race conditions)
        __syncthreads();
    }
    
    // 3. Store Phase
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =========================================================================
// 2. Main Function (Host Code)
// =========================================================================

int main()
{
    // Define matrix sizes (must be compatible with TILE=32 for efficiency)
    int M = 64, N = 64, K = 64; // Using larger, tile-aligned sizes

    // Adjust sizes if they were smaller than TILE
    if (M < TILE || N < TILE || K < TILE) {
        cout << "Warning: Matrix dimensions are smaller than TILE size (32). Performance may be sub-optimal." << endl;
        // Reverting to user's original dimensions for compliance, but showing the launch config
        M = 2; N = 3; K = 3;
    }
    
    // Host memory pointers
    int *arr, *brr, *crr;

    // Allocate Host Memory
    arr = (int*) malloc(M * K * sizeof(int));
    brr = (int*) malloc(K * N * sizeof(int));
    crr = (int*) malloc(M * N * sizeof(int));

    if(!arr || !brr || !crr)
    {
        cerr << "malloc needs to check" << endl;
        exit(EXIT_FAILURE);
    }

    // Initialize Matrices A and B
    fill_random(arr, M ,K);
    fill_random(brr, K ,N);
    printf("Matrix A (%d x %d):\n", M, K);
    print_matrix(arr,M,K);
    printf("\nMatrix B (%d x %d):\n", K, N);
    print_matrix(brr,K,N);

    // Device memory pointers
    int *d_a, *d_b, *d_c;

    // Allocate Device Memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, M * K *sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, K * N *sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, M * N *sizeof(int)));

    // Copy Host to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, arr, M * K * sizeof(int) , cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, brr, K * N * sizeof(int) , cudaMemcpyHostToDevice));

    // --- Launch Configuration for Tiled Kernel ---
    // Grid dimensions must cover the M x N output matrix using TILE x TILE blocks
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE); // grid.x covers N, grid.y covers M
    
    cout << "\nLaunching Tiled Kernel with Grid (" << grid.x << "," << grid.y << ") and Block (" << block.x << "," << block.y << ")" << endl;

    // Launch Tiled Kernel
    f_mat_mul_tiled<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    // Check for errors and synchronize
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy Result Device to Host
    CHECK_CUDA_ERROR(cudaMemcpy(crr, d_c, M * N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\nMatrix C (%d x %d) (Tiled Result):\n", M, N);
    print_matrix(crr,M,N);

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));

    free(arr);
    free(brr);
    free(crr);

    return 0;
}
