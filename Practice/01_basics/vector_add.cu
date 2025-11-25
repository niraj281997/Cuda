#include<iostream>
#include<stdlib.h>
#include<cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

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

__global__ void f_vector_add(float *a , float *b , float *c, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx<N)
	{
		c[idx] = a[idx] + b[idx];
	}

}


int main()
{
	cudaError_t err = cudaSuccess;
	
	int numofelement = 256;
	size_t size = numofelement*sizeof(float);

	printf("[vector addition %d elements]\n",numofelement);



	float *h_a = (float*) malloc(size);
	float *h_b = (float*) malloc(size);
	float *h_c = (float*) malloc(size);

	if(h_a == NULL || h_b == NULL || h_c == NULL)
	{

		fprintf(stderr,"Failed to allocate host vectors\n");
		exit(EXIT_FAILURE);
	}

	for(int i = 0 ; i<numofelement ; i++)
	{

		h_a[i] = rand() /(float) RAND_MAX;
		h_b[i] = rand() /(float) RAND_MAX;
	}

	float *d_a = NULL;
	float *d_b = NULL;
	float *d_c = NULL;


	CHECK_CUDA_ERROR(cudaMalloc(&d_a,size));
	CHECK_CUDA_ERROR(cudaMalloc(&d_b,size));
	CHECK_CUDA_ERROR(cudaMalloc(&d_c,size));
	
	printf("Copying input data from the host memory to the CUDA device\n");

	CHECK_CUDA_ERROR(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));

	int threadPerBlock = 32;
	int blockPerGrid = (numofelement + threadPerBlock - 1)/threadPerBlock;

	f_vector_add<<<threadPerBlock,blockPerGrid>>>(d_a,d_b,d_c,numofelement);
	
	CHECK_CUDA_ERROR(cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost));


	for(int i = 0 ; i < numofelement ; i++)
    	{
       		printf("[%d] = %d\n", i ,h_c[i]);
       	}


	CHECK_CUDA_ERROR(cudaFree(d_a));
	CHECK_CUDA_ERROR(cudaFree(d_b));
	CHECK_CUDA_ERROR(cudaFree(d_c));
	

	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
