#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>

using namespace std;

// --- CUDA Parameters ---
#define TILE 8                  // Block dimension (e.g., 32x32 threads)
#define TILE_PAD (TILE + 1)      // Padding to prevent shared memory bank conflicts


#define CHECK_CUDA_ERROR(val) check(val, #val, __FILE__, __LINE__)

template <class T>
void check(T err , const char *func , const char * file, const int  line )
{

	if(err != cudaSuccess)
	{
		cerr << "CUDA Error at " << file << ":" << line << " - " << cudaGetErrorString(err) << " [" << func << "]\n";
        	exit(EXIT_FAILURE);
	}
}

//Utility function to print a matrix subset.
void print_matrix_subset(int R, int C, const int* M, const string& name, int count = 4) {
    cout << "\n--- " << name << " (" << R << "x" << C << ") (Subset) ---\n";
    for (int i = 0; i < min(R, count); i++) {
        for (int j = 0; j < min(C, count); j++) {
            cout << std::setw(10) << M[i * C + j];
        }
        if (C > count) cout << " ...";
        cout << endl;
    }
    if (R > count) cout << "...\n";
    cout << "---------------------------------\n";
}
__global__ void f_Matrix_Transpose(const int * A, int *B, const int R, const int C)
{

	__shared__ int tile [TILE][TILE_PAD];


	int global_x_read = blockIdx.x * TILE +  threadIdx.x;
	int global_y_read= blockIdx.y * TILE + threadIdx.y;


	int global_x_write = blockIdx.y * TILE + threadIdx.x;
	int global_y_write = blockIdx.x * TILE + threadIdx.y;

	if(global_x_read < R && global_y_read < C)
	{
		int index_A = global_y_read * C + global_x_read;
		tile[threadIdx.x][threadIdx.y] = A[index_A];
	}
	__syncthreads();

	if(global_x_write < R && global_y_write < C)	
	{
		int index_B = global_x_write * R + global_y_write;
		B[index_B] = tile[threadIdx.x][threadIdx.y];
	}


}

int main()
{
	const int R = 32;
	const int C = 32;

	int size = R * C * sizeof(int);


	int * h_a =(int*) malloc(size);
	int * h_b =(int*) malloc(size);
	
	if(h_a ==NULL ||  h_b ==NULL)
	{
		fprintf(stderr, "Malloc allocation has some issue\n");
		exit(EXIT_FAILURE);
	}
	for(int i = 0; i < R * C; ++i) {
        h_a[i] = i;
    }

	dim3 block(TILE,TILE);
	dim3 grid( (R + TILE -1 )/ TILE,\
			(C + TILE -1 )/TILE);
	
	int *d_a, *d_b;
	

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a,size));	
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b,size));	
	CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
	
	cout << "[Tiled Matrix Transpose " << R << "x" << C << " elements]\n";
    	cout << "Launching kernel with grid (" << grid.x << "," << grid.y << ") and block (" << block.x << "," << block.y << ")\n";

    
	f_Matrix_Transpose<<<grid,block>>>(d_a,d_b,R,C);
	
	CHECK_CUDA_ERROR(cudaGetLastError());

	CHECK_CUDA_ERROR(cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost));

	print_matrix_subset(R, C, h_a, "Original Matrix A");
	print_matrix_subset(C, R, h_b, "Transposed Matrix B");
	free(h_a);
	free(h_b);



}
