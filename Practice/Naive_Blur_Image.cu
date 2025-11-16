#include<iostream>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<vector>
using namespace std;

__device__ __forceinline__ int f_check_range(int v, int low , int high)
{

	return  (v< low) ? low : (v > high) ? high : v;
}
__global__ void f_naive_blur_box_kernel(const unsigned char * in, unsigned char * out, int height, int width, int radius)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	

	if(x>width || y>height) return ;
	
		int kernalsize = 2*radius +1;
	int kernalarea = kernalsize * kernalsize;
		int sum = 0;

	for(int ky = -radius ; 	ky<= radius ; ky++)
	{

		int sy = f_check_range( y + ky , 0 , height -1);

		for(int kx = - radius; kx <= radius ; kx ++)
		{
			int sx = f_check_range( x + kx , 0 , width -1 );
			sum += in[sy * width + sx];

		}
	}
	out[y*width + x] = (unsigned char)sum/kernalarea;
}

int main()
{
	int height = 1024;
	int width = 768;
	int radius = 1;

	int bytes = height * width;

	vector<unsigned char> img(bytes),out(bytes);

	for(int i= 0 ; i< bytes ; i++)
	{
		img[i] = i %256;
	}
	unsigned char *d_in, *d_out;


	cudaMalloc(&d_in,bytes);
	cudaMalloc(&d_out,bytes);


	cudaMemcpy(d_in,img.data(),bytes,cudaMemcpyHostToDevice);

	dim3 block(16,16);
	dim3 grid((bytes+ block.x -1)/block.x,
			(bytes+block.y-1)/block.y);

	f_naive_blur_box_kernel<<<grid,block>>>(d_in,d_out,height,width,radius);

	cudaDeviceSynchronize();
	
	cudaMemcpy(out.data(),d_out,bytes,cudaMemcpyDeviceToHost);

	std::cout<<"Blurcomplete"<<endl;

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
