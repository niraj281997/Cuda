#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include<cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

using namespace std;

template <class T >
void check(T err, const char* const func, const char * const file,  const int line )
{
	if(err != cudaSuccess)
	{
		fprintf(stderr," CUDA Runtime Error at : %s:%d\n",file,line);
		fprintf(stderr,"%s %s\n",cudaGetErrorString(err),func);
		exit(EXIT_FAILURE);
	}
}



__global__ void f_gpu_Matrix_Transpose(const float *a , float *b , const int R, const int C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx< R && idy < C)
	{
		int a_index = idx * C + idy;
		int b_index = idy *R + idx;

		b[b_index] = a[a_index];
	}

}


void print_matrix(int R, int C, const float* M, const string& name) {
    cout << "\n--- " << name << " (" << R << "x" << C << ") ---\n";
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            cout << fixed << setprecision(1) << setw(6) << M[i * C + j];
        }
        cout << endl;
    }
    cout << "---------------------------------\n";
}



void f_GPU_config(const float *h_a, float *h_b, const int size, const int R, const int C)
{

	float *d_a = NULL;
	float *d_b = NULL;

	CHECK_CUDA_ERROR(cudaMalloc(&d_a,size));
	CHECK_CUDA_ERROR(cudaMalloc(&d_b,size));
	
	printf("Copying input data from the host memory to the CUDA device\n");

	CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));



	int thread_x= 16;
	int thread_y= 16;

	dim3 block(thread_x,thread_y);
	dim3 grid((R+ thread_x -1 )/thread_x, \
			(C + thread_y -1 )/thread_y);


	f_gpu_Matrix_Transpose<<<grid,block>>>(d_a,d_b,R,C);
	
	CHECK_CUDA_ERROR(cudaMemcpy(h_b,d_b,size,cudaMemcpyDeviceToHost));


	print_matrix(R,C, h_a, "Original Matrix");
	print_matrix(R,C, h_b, "Transpose Matrix");

	CHECK_CUDA_ERROR(cudaFree(d_a));
	CHECK_CUDA_ERROR(cudaFree(d_b));
	
}


void f_Naive_Transpose_CPU(const float * h_a, float * h_b, const int R, const int C)
{
	for (int i = 0 ; i< R ; i++)
	{
		for(int j = 0 ; j < C; j++)
		{
			int A_index = i *C + j;  // Row major access 
			int B_index = j * R + i; // column mejor access;

			h_b[B_index]= h_a[A_index];
		}
	}

	return;
}
int main()
{
	
	int R = 5;
	int C = 5;
	size_t size = R*C*sizeof(float);

	printf("[vector addition %d x %d elements]\n",R,C);



	float *h_a = (float*) malloc(size);
	float *h_b = (float*) malloc(size);

	if(h_a == NULL || h_b == NULL)
	{

		fprintf(stderr,"Failed to allocate host vectors\n");
		exit(EXIT_FAILURE);
	}

	for(int i = 0 ; i<R*C ; i++)
	{

		h_a[i] = (float)(rand() %10);

	}

	
/*	
	print_matrix(R,C,h_a,"Original Matrix");
	f_Naive_Transpose_CPU(h_a,h_b,R,C);
	print_matrix(R,C,h_b,"Transpose Matrix");
*/
	f_GPU_config(h_a,h_b,size,R,C);


	free(h_a);
	free(h_b);

	return 0;
}
