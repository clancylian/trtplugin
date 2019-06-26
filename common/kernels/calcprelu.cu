/**
 * @file       calcprelu.cu
 * @brief      实现插件PReLU层的CUDA核函数
 * @details    实现插件PReLU层的CUDA核函数
 * @author     clancy.lian@gmail.com
 * @date       2017.12.25
 * @version    V0.1
 * @par Copyright (C):
 *			   罗普特(厦门)科技集团有限公司
 * @par History:
 *  -V0.1      clancy.lian@gmail.com       2017.12.25 \n
 *             原型开发 \n
 */

#include <cuda_runtime.h>

__global__ void calcPReLUKernel(const float *input, float *output, const float *weights,
                          int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    output[y * width + x] = input[y * width + x] > 0 ? input[y * width + x] : input[y * width + x] * weights[y % channels];

}

void calcPReLU(const float *input, float *output, const float* weights, int batchSize, int channels,
                          int width, int height, cudaStream_t stream)
{
    dim3 grids((width * height + 31) / 32, (batchSize * channels + 31) / 32);
    dim3 blocks(32, 32);
    calcPReLUKernel<<<grids, blocks, 0, stream>>>(input, output, weights, width * height, channels * batchSize, channels);
}
