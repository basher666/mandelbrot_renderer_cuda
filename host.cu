#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mandelbrot_kernel(int*, int*, double, double, double, double, int);

int main()
{
	cudaError_t err = cudaSuccess;
	int dim_x = 1024;
	int dim_y = 1024;
	double l_margin = -2.0;
	double r_margin = 1.0;
	double u_margin = 1.5;
	double d_margin = -1.5;
	int N = 64;
	dim3 threads_per_block(32,32,1);
	dim3 blocks_per_grid(32,32,1);
	
	size_t canvas_size =  dim_x*dim_y*sizeof(int);

	int *h_canvas = (int *)malloc(sizeof(int)*dim_x*dim_y);
	memset(h_canvas, 0, dim_x*dim_y*sizeof(int));

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

	printf("Getting the canvas back from kernel\n");
	fflush(stdout);

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

	printf("Freeing device memory\n");
	fflush(stdout);
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


	printf("Writing the rendered image\n");
	fflush(stdout);
	FILE* img = fopen("mandelbrot.pgm", "wb");
	(void )(void) fprintf(img, "P6\n%d %d\n255\n", dim_x, dim_y);
	printf("Writing RGB values ...\n");
	fflush(stdout);

	for(int i=0;i<dim_y;i++)
	{
		for(int j=0;j<dim_x;j++)
		{
			unsigned char tmp[3];
			tmp[0] = 0;
			tmp[1] = (char)h_canvas[i*dim_x + j];
			tmp[2] = 0;
			(void) fwrite(tmp, 1, 3, img);
		}
	}
	if (img==NULL)
		printf("img is NULL\n");
	(void) fclose(img);
	free(h_canvas);
}