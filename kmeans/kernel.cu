
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Header.h"

#define MAX_THREAD_PER_BLOCK 1000
#define MAX_NUM_OF_WORKS_PER_THREAD 100
#define MIN_NUM_OF_WORKS_PER_THREAD 1

//each thred promotes the points he is responsible for
__global__ void changeFrameKernel(Point* arrPoint, const double dT,const int startIndex, const int num_of_works_per_thread, const int num_of_threads_per_block)
{
	int startThreadIndex = startIndex+ (threadIdx.x + blockIdx.x * num_of_threads_per_block)*num_of_works_per_thread;
	int endThreadIndex = startThreadIndex + num_of_works_per_thread;
	// newLocation = currentLoc + (dT*v)
	for (int i = startThreadIndex; i < endThreadIndex; i++) {
		arrPoint[i].x = arrPoint[i].x + dT*arrPoint[i].vx;
		arrPoint[i].y = arrPoint[i].y + dT*arrPoint[i].vy;
	}
}


// initial cuda and run kernel for change point in cuda
cudaError_t changeFrameStart(Point* arrPoint, const int n, const double dT,Point **point_to_dev_arrPoint)
{
	//every block have 1000 threads and every thread have 100 tasks
	const int fullJobsThreads = n / MAX_NUM_OF_WORKS_PER_THREAD;
	const int full_Jobs_blocks_in_grid = fullJobsThreads / MAX_THREAD_PER_BLOCK; 
	//rest tasks
	const int rest_Threads = fullJobsThreads- full_Jobs_blocks_in_grid*MAX_THREAD_PER_BLOCK; // threads with "MAX_NUM_OF_WORKS_PER_THREAD" tasks for 1 last block
	const int restJobs = n - (fullJobsThreads*MAX_NUM_OF_WORKS_PER_THREAD); //Rest from div less then  MAX_NUM_OF_WORKS_PER_THREAD for 1 thread

	int startIndex = 0;
	Point *dev_arrPoint = 0;
	cudaError_t cudaStatus;
	int errorFlag = 0;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		errorFlag = 1;
		goto Error;
	}
	// Allocate GPU buffers for three arrays 
	cudaStatus = cudaMalloc((void**)&dev_arrPoint, n * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_arrPoint failed!");
		errorFlag = 1;
		goto Error;
	}
	// Copy input arrays from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_arrPoint, arrPoint, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy arrpoint to dev_arrPoint failed!");
		errorFlag = 1;
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.
	if(full_Jobs_blocks_in_grid!=0)
		changeFrameKernel << <full_Jobs_blocks_in_grid, MAX_THREAD_PER_BLOCK >> >(dev_arrPoint,dT,startIndex, MAX_NUM_OF_WORKS_PER_THREAD, MAX_THREAD_PER_BLOCK);
	startIndex += MAX_NUM_OF_WORKS_PER_THREAD * (fullJobsThreads-rest_Threads);
	if (rest_Threads != 0)
	changeFrameKernel << <1, rest_Threads >> >(dev_arrPoint, dT,startIndex, MAX_NUM_OF_WORKS_PER_THREAD, rest_Threads);
	startIndex += MAX_NUM_OF_WORKS_PER_THREAD * rest_Threads;
	changeFrameKernel << <1, restJobs >> >(dev_arrPoint, dT, startIndex, MIN_NUM_OF_WORKS_PER_THREAD,restJobs);

	*point_to_dev_arrPoint = dev_arrPoint;
	
	Error:
		if(errorFlag == 1)
			cudaFree(dev_arrPoint);
	
	return cudaStatus;
}

// get results of new point arr from gpu
cudaError_t changeFrameFinish(Point* arrPoint, const int n, cudaError_t cudaStatus, Point **point_to_dev_arrPoint)
{
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching changeFrameFinish!\n", cudaStatus);
		goto Error;
	}

	// Copy output total count array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arrPoint,*point_to_dev_arrPoint, n * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy changeFrameFinish failed!");
		goto Error;
	}
Error:
	cudaFree(*point_to_dev_arrPoint);

	return cudaStatus;
}