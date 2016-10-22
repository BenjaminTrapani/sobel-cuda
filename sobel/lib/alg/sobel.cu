/* Kernel by Benjamin Trapani
 * Setup by Julian Gutierrez
 * Northeastern University
 * High Performance Computing
 * 
 * Sobel Algorithm Implementation 
 *  
 */
 
#include "sobel.h"

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

using namespace std;

void modThreshold (unsigned int value){
	threshold = value;
}

/*
 * Sobel Kernel
 */


__device__ void getZOrderIndex(unsigned int x, unsigned int y, unsigned int * result) {
	*result = y * (BLOCK_TILE_SIZE + 2) + x;
}

__device__ void fillSharedMemory(const unsigned int tx, 
const unsigned int ty,
const unsigned int x,
const unsigned int y,
const unsigned int xsize, 
const unsigned short middleRow,
const unsigned int rowOffset,
unsigned char * intensity,
unsigned char * valuesInBlock) {
        const unsigned int yUp = ty;
        const unsigned int yMid = ty+1;
        const unsigned int yLow = ty+2;

        const unsigned int xLeft = tx;
        const unsigned int xMid = tx + 1;
        const unsigned int xRight = tx + 2;
	
	unsigned int v1, v2, v3, v4, v5, v6, v8, v11, vCenter;
	getZOrderIndex(xRight, yUp, &v1);
	getZOrderIndex(xLeft, yUp, &v2);
	getZOrderIndex(xRight, yMid, &v3);
	getZOrderIndex(xLeft, yMid, &v4);
	getZOrderIndex(xRight, yLow, &v5);
	getZOrderIndex(xLeft, yLow, &v6);
	getZOrderIndex(xMid, yUp, &v8);
	getZOrderIndex(xMid, yLow, &v11);
	getZOrderIndex(xMid, yMid, &vCenter);
	valuesInBlock[vCenter] = (unsigned char)((middleRow >> (rowOffset * 8)));
		const unsigned int upperBound = (BLOCK_TILE_SIZE-1);
                if (tx == 0) {
                        if (ty == 0) {
                                valuesInBlock[v2] = intensity[ xsize * (y-1) + x-1 ];
                        }
                        valuesInBlock[v4] = intensity[ xsize * (y)   + x-1 ];
                        if (ty == upperBound) {
                                valuesInBlock[v6] = intensity[ xsize * (y+1) + x-1 ];
                        }
               }
               if (tx == upperBound) {
                        if (ty == 0) {
                                valuesInBlock[v1] = intensity[ xsize * (y-1) + x+1 ];
                        }
                        valuesInBlock[v3] = intensity[ xsize * (y)   + x+1 ];
                        if (ty == upperBound) {
                                valuesInBlock[v5] = intensity[ xsize * (y+1) + x+1 ];
                        }
              }
              if (ty == 0) {
                        valuesInBlock[v8] = intensity[ xsize * (y-1) + x   ];
              }
              if (ty == upperBound) {
                        valuesInBlock[v11] = intensity[ xsize * (y+1) + x   ];
              }
}

__device__ void performComputation(const unsigned int tx, 
const unsigned int ty,
const unsigned int x,
const unsigned int y,
const unsigned int xsize,
unsigned char * valuesInBlock,
unsigned int threshold,
unsigned char *result) {
	const unsigned int yUp = ty;
        const unsigned int yMid = ty+1;
        const unsigned int yLow = ty+2;

        const unsigned int xLeft = tx;
        const unsigned int xMid = tx + 1;
        const unsigned int xRight = tx + 2;

        unsigned int v1, v2, v3, v4, v5, v6, v8, v11;
                getZOrderIndex(xRight, yUp, &v1);
                getZOrderIndex(xLeft, yUp, &v2);
                getZOrderIndex(xRight, yMid, &v3);
                getZOrderIndex(xLeft, yMid, &v4);
                getZOrderIndex(xRight, yLow, &v5);
                getZOrderIndex(xLeft, yLow, &v6);
                getZOrderIndex(xMid, yUp, &v8);
                getZOrderIndex(xMid, yLow, &v11);
	
		const int resultLocation = y * xsize + x;
                unsigned char v1Val, v2Val, v5Val, v6Val;
                v1Val = valuesInBlock[v1];
                v2Val = valuesInBlock[v2];
                v5Val = valuesInBlock[v5];
                v6Val = valuesInBlock[v6];
                int sum1 = v1Val -
                        v2Val +
                    2 * valuesInBlock[v3] -
                    2 * valuesInBlock[v4] +
                        v5Val -
                        v6Val;

                int sum2 = v2Val +
                       2 * valuesInBlock[v8] +
                           v1Val -
                           v6Val -
                       2 * valuesInBlock[v11] -
                           v5Val;
                int magnitude = sum1*sum1 + sum2*sum2;
                if (magnitude > threshold)
                        result[resultLocation] = 255;
                else
                        result[resultLocation] = 0;
}

__global__ void sobelAlgorithm(unsigned char *intensity, 
				unsigned char *result,
				unsigned int threshold){
        
        int tx = threadIdx.x;
	int ty = threadIdx.y;
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
        
        int xsize = TILE_SIZE*gridDim.x;
        int ysize = TILE_SIZE*gridDim.y;
        
	extern __shared__ unsigned char valuesInBlock[];
	const int xBase = threadIdx.x * 2;
	int x = bx*TILE_SIZE+xBase;
	int y = by*TILE_SIZE+ty;
	
	const bool shouldFillSharedMem = (y > 0 && y < ysize && x >= 0 && x < xsize-1);
	const int xIterations = 2; 
	if (shouldFillSharedMem) {
		unsigned short * intIntensities = (unsigned short*)intensity;
		unsigned int rowValue = 0;
		rowValue = intIntensities[((xsize * y) + x) / 2];
		#pragma unroll
		for(tx = xBase; tx < xBase + xIterations; tx++) {
			x = bx*TILE_SIZE+tx;
			fillSharedMemory(tx, threadIdx.y, x, y, xsize, rowValue, tx - xBase, intensity, valuesInBlock);
		} 
	}
	
	__syncthreads();
	
	const bool pointWithinOutsideBorder = (y > 1 && y < ysize-1 && x > 1 && x < xsize-2);
        if (pointWithinOutsideBorder) {
		#pragma unroll
		for(tx = xBase; tx < xBase + xIterations; tx++) {
                        x = bx*TILE_SIZE+tx;
                        performComputation(tx, threadIdx.y, x, y, xsize, valuesInBlock, threshold, result);
                }
	}
}

unsigned char *sobel(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){
	
	#if defined(DEBUG)
		printf("Printing input data\n");
		printf("Height: %d\n", height);
		printf("Width: %d\n", width);
	#endif
	
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	#if defined(VERBOSE)
		printf ("Allocating arrays in GPU memory.\n");
	#endif
	
	#if defined(CUDA_TIMING)
		float Ttime;
		TIMER_CREATE(Ttime);
		TIMER_START(Ttime);
	#endif
	
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
	
	#if defined(VERBOSE)
		printf("Running algorithm on GPU.\n");
	#endif
	
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE/2, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
	sobelAlgorithm<<<dimGrid, dimBlock, (BLOCK_TILE_SIZE+2) * (BLOCK_TILE_SIZE+2) * 4 * sizeof(unsigned char)>>>(gpu.intensity, 
					      			       gpu.result,
					      			       threshold);

	checkCuda(cudaDeviceSynchronize());

	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
	
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ttime);
		printf("Total GPU Execution Time: %f ms\n", Ttime);
	#endif
	
	return(gpu.resultOnCPU);

}

unsigned char *sobelWarmup(unsigned char *intensity,
		unsigned int height, 
		unsigned int width){

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	
	// Both are the same size (CPU/GPU).
	gpu.size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&gpu.intensity              , gpu.size*sizeof(char)));
	checkCuda(cudaMalloc((void**)&gpu.result                 , gpu.size*sizeof(char)));
	
	// Allocate result array in CPU memory
	gpu.resultOnCPU = new unsigned char[gpu.size];
				
        checkCuda(cudaMemcpy(gpu.intensity, 
			intensity, 
			gpu.size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_TILE_SIZE/2, BLOCK_TILE_SIZE);
	
	// Launch kernel to begin image segmenation
        sobelAlgorithm<<<dimGrid, dimBlock,  (BLOCK_TILE_SIZE + 2) * (BLOCK_TILE_SIZE + 2) * 4 * sizeof(unsigned char)>>>(gpu.intensity,
                                                                       gpu.result,
                                                                       threshold);
	
	checkCuda(cudaDeviceSynchronize());

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(gpu.resultOnCPU, 
			gpu.result, 
			gpu.size*sizeof(char), 
			cudaMemcpyDeviceToHost));
			
	// Free resources and end the program
	checkCuda(cudaFree(gpu.intensity));
	checkCuda(cudaFree(gpu.result));
	
	return(gpu.resultOnCPU);

}
