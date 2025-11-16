#include<iostream>
#include<cuda_runtime.h>
#include<stdlib.h>

__global__ void f_function(const int *a,const int *b, int *c)
{
	int Idx = threadIdx.x + blockIdx.x * blockDim.x;
	c[Idx] = a[Idx] + b[Idx];
	return;
}

int main()
{

	int *h_a,*h_b,*h_c;
	int *d_a,*d_b,*d_c;
	int N = 1<<16;
	int size = N * sizeof(int);

	int threads= 1024;
	int blocks = (N+threads-1)/threads;
	dim3 block(blocks);
	dim3 thread(threads);
	
	cudaEvent_t start, stop;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start);

	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);


	for(int i = 0 ; i< N ; i++)
	{
		h_a[i] = i;
		h_b[i] = i *2;
	}
	cudaMalloc(&d_a,size);
	cudaMalloc(&d_b,size);
	cudaMalloc(&d_c,size);
	
	cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

	f_function<<<block,thread>>>(d_a,d_b,d_c);

	cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds;

	cudaEventElapsedTime(&milliseconds,start,stop);

	for(int i = 0 ; i< N ; i++)
	{
		std::cout<<h_a[i]<<"+"<<h_b[i]<<"="<<h_c[i]<<std::endl;
	}
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";



	return 0;
}
