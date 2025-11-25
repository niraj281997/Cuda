#include<iostream>
#include<cuda_runtime.h>

using namespace std;

#define CHECK_CUDA_ERROR(val) check(val, #val, __FILE__ , __LINE__)

#define TILE 32

template <class T>
void check(T err , const char * func, const char * file , const int line)
{
	if(err!=cudaSuccess)
	{

		exit(0);
	}
}

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
__global__ void f_mat_mul(int * A, int * B, int *C, const int M ,const int N , const int K)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	int sum = 0;
	if(row < M && col < N )
	{
		for(int i = 0 ; i < K ; i ++)
		{
			sum += A[ row * K + i] * B[ i * N + col];
		}
	}
	C[row * N + col] = sum;

}

int main()
{

	int M = 2, N = 3 , K = 3;
	int *d_a, *d_b, *d_c;

	
	dim3 block(TILE,TILE);
	dim3 grid((block.x + TILE -1 )/TILE, (block.y + TILE -1)/TILE);

	int *arr =(int*) malloc(M * K * sizeof(int));
	int *brr =(int*) malloc(K * N * sizeof(int));
	int *crr =(int*) malloc(M * N * sizeof(int));

	

	if(!arr || !brr || !crr)
	{
		cerr<<"malloc needs to check"<<endl;
	}

	fill_random(arr, M ,K);
	fill_random(brr, K ,N);
	printf("Matrix A (%d x %d):\n", M, K);
	print_matrix(arr,M,K);
	printf("\nMatrix B (%d x %d):\n", K, N);
	print_matrix(brr,K,N);

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a,M * K *sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b,K * N *sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c,M * N *sizeof(int)));
	

	CHECK_CUDA_ERROR(cudaMemcpy(d_a, arr, M * K * sizeof(int) , cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_b, brr, K * N * sizeof(int) , cudaMemcpyHostToDevice));
	
	f_mat_mul<<<grid,block>>>(d_a,d_b,d_c, M , N ,K);
	
	CHECK_CUDA_ERROR(cudaMemcpy(crr, d_c, M * N * sizeof(int), cudaMemcpyDeviceToHost));
	
	printf("\nMatrix C (%d x %d):\n", M, N);
	print_matrix(crr,M,N);


	CHECK_CUDA_ERROR(cudaFree(d_a));
	CHECK_CUDA_ERROR(cudaFree(d_b));
	CHECK_CUDA_ERROR(cudaFree(d_c));


	free(arr);
	free(brr);
	free(crr);


	return 0;
}
