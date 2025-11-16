#include <iostream>
#include <cuda_runtime.h>
using namespace std;

/* stb libraries */
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

__device__ __forceinline__
int clamp(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

__global__
void blur_kernel(const unsigned char* in,
                 unsigned char* out,
                 int width,
                 int height,
                 int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int size = 2 * radius + 1;
    int area = size * size;
    int sum = 0;

    for (int ky = -radius; ky <= radius; ++ky) {
        int sy = clamp(y + ky, 0, height - 1);

        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = clamp(x + kx, 0, width - 1);
            sum += in[sy * width + sx];
        }
    }

    out[y * width + x] = sum / area;
}

int main()
{
    int width, height, channels;

    // Load grayscale image (force 1 channel)
    unsigned char* img = stbi_load("../Sample.png",
                                   &width,
                                   &height,
                                   &channels,
                                   1);

    if (!img) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    cout << "Loaded " << width << "x" << height << endl;

    int bytes = width * height;  // grayscale = 1 byte per pixel

    unsigned char *d_in, *d_out;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, img, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    int radius = 1;

    blur_kernel<<<grid, block>>>(d_in, d_out, width, height, radius);
    cudaDeviceSynchronize();

    unsigned char* out = new unsigned char[bytes];

    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    // write output PNG
    stbi_write_png("output.png",
                   width,
                   height,
                   1,
                   out,
                   width);

    cout << "Blur complete" << endl;

    stbi_image_free(img);
    delete[] out;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

