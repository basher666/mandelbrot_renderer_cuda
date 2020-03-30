#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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
		canvas[idx] = (int)(((double)escape_time/(double) N)*255.0);
	num_it[idx] = escape_time;
}