#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "kernel.h"

using namespace std;
int main()
{
	double center_x = -0.73985494;
	double center_y = 0.16362172;
	printf("Centre : %lf + i %lf \n",center_x,center_y);
	fflush(stdout);
	int dim_x = 1024;
	int dim_y = 1024;
	size_t canvas_size =  dim_x*dim_y*sizeof(int);
	int it, init_len;
	double zoom;

	int *h_canvas = NULL;
	std::string filename = "frames/mandelbrot";
	for (it=0, zoom=1.0, init_len = 2.0; zoom<=pow(10,40); zoom *= 1.1, it++)
	{
		string it_s = to_string(it);
		h_canvas = (int *)malloc(canvas_size);
		render(h_canvas, center_x, center_y, init_len/zoom, dim_x, dim_y);

		// printf("Writing the rendered image\n");
		// fflush(stdout);
		FILE* img = fopen((filename + it_s + string(".pgm")).c_str(), "wb");
		(void )(void) fprintf(img, "P6\n%d %d\n255\n", dim_x, dim_y);
		// printf("Writing RGB values ...\n");
		// fflush(stdout);

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
		free(h_canvas);
		(void) fclose(img);
	}
}