#ifndef mandelbrot
#define mandelbrot

void mandelbrot_kernel(int *canvas, int *num_it, double l_margin, double r_margin, double u_margin, double d_margin, int N);
void render(double , double , double , double );

#endif