#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelbrot_kernel(int *canvas, int *num_it, double l_margin, double r_margin, double u_margin, double d_margin, int N)
{
	int num_rows = blockDim.y*gridDim.y;
	int num_cols = blockDim.x*gridDim.x;
	double z_n_x = 0;
	double z_n_y = 0;
	double tmp_x, tmp_y;
	int tid_x = blockIdx.x*blockDim.x + threadIdx.x;
	int tid_y = blockIdx.y*blockDim.y + threadIdx.y;
	double c_x = l_margin + (tid_x/(double)(num_cols -1))*(r_margin - l_margin);
	double c_y = d_margin + (tid_y/(double)(num_rows -1))*(u_margin - d_margin);
	int escape_time = 0;
	int idx = tid_y*num_cols + tid_x;

	while(z_n_x*z_n_x + z_n_y*z_n_y < 4 && escape_time<N)
	{
		tmp_x = z_n_x*z_n_x - z_n_y*z_n_y;
		tmp_y = 2*z_n_x*z_n_y;
		z_n_x = tmp_x + c_x;
		z_n_y = tmp_y + c_y;
		escape_time ++;
	}
	if (escape_time==N)
		canvas[idx] = 0;
	else
	{
		double mod = z_n_x*z_n_x + z_n_y*z_n_y;
		canvas[idx] = (int)(((escape_time - log(log(mod))/log(2.0))/(double)N)*255.0);
		// canvas[idx] = (int)(((double)escape_time/(double) N)*255.0);
	}
	num_it[idx] = escape_time;
}

void render(int *h_canvas,long double center_x,long double center_y, double init_len, int dim_x, int dim_y)
{
	cudaError_t err = cudaSuccess;
	
	double l_margin = center_x - init_len/2.0;
	double r_margin = center_x + init_len;
	double u_margin = center_y + init_len/2.0;
	double d_margin = center_y - init_len/2.0;
	int N = 255;
	dim3 threads_per_block(32,32,1);
	dim3 blocks_per_grid(dim_x/32,dim_y/32,1);
	
	size_t canvas_size =  dim_x*dim_y*sizeof(int);

	

	int *h_num_it = (int*)malloc(canvas_size);
	memset(h_num_it, 0, canvas_size);

	int *d_canvas = NULL;
	err = cudaMalloc((void **)&d_canvas, canvas_size);
	if(err != cudaSuccess)
	{
		printf("Error in cudaMalloc : d_canvas\n");
		exit(EXIT_FAILURE);
	}

	int *d_num_it = NULL;
	err = cudaMalloc((void **)&d_num_it, canvas_size);
	if(err != cudaSuccess)
	{
		printf("Error in cudaMalloc : d_num_it\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_canvas, h_canvas, canvas_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Error in cudaMemcpy : d_canvas\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_num_it, h_num_it, canvas_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Error in cudaMemcpy : d_num_it\n");
		exit(EXIT_FAILURE);
	}

	mandelbrot_kernel <<<blocks_per_grid, threads_per_block>>> (d_canvas, d_num_it, l_margin, r_margin, u_margin, d_margin, N);

	err = cudaGetLastError();
	if(err!=cudaSuccess)
	{
		printf("Error in kernel\n");
		exit(EXIT_FAILURE);
	}

	// printf("Getting the canvas back from kernel\n");
	// fflush(stdout);

	err = cudaMemcpy(h_canvas, d_canvas, canvas_size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess)
	{
		printf("Error in cudaMemcpy: h_canvas\n");
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(h_num_it, d_num_it, canvas_size, cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess)
	{
		printf("Error in cudaMemcpy: h_num_it\n");
		exit(EXIT_FAILURE);
	}

	// printf("Freeing device memory\n");
	// fflush(stdout);
	err = cudaFree(d_canvas);
	if(err!=cudaSuccess)
	{
		printf("Error in cudaFree: d_canvas\n");
		exit(EXIT_FAILURE);
	}

	err = cudaDeviceReset();
	if(err!=cudaSuccess)
	{
		printf("Error in cudaDeviceReset\n");
		exit(EXIT_FAILURE);
	}	

	// printf("analyzing escape times\n");
	// fflush(stdout);
	
	// int max_esc = 0;
	// int min_esc = 1000;
	// double avg_esc = 0.0;
	// int outside_count = 0;
	// for(int i=0;i<dim_y;i++)
	// {
	// 	for(int j=0;j<dim_x;j++)
	// 	{
	// 		if (h_canvas[i*dim_x + j]!=0)
	// 		{
	// 			if (h_num_it[i*dim_x + j]>max_esc)
	// 				max_esc = h_num_it[i*dim_x + j];
	// 			if (h_num_it[i*dim_x + j]<min_esc)
	// 				min_esc = h_num_it[i*dim_x + j];
	// 			avg_esc += h_num_it[i*dim_x + j];
	// 			outside_count ++;
	// 		}
	// 	}
	// }
	// avg_esc = avg_esc/outside_count;
	// printf("max it:%d, min_it:%d, outside_count:%d, avg_it:%lf",max_esc, min_esc, outside_count, avg_esc);


	
}