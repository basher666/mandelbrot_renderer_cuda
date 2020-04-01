#ifndef mandelbrot
#define mandelbrot

void mandelbrot_kernel(int *canvas, int *num_it, double l_margin, double r_margin, double u_margin, double d_margin, int N);
void render(int *,long double ,long double , double , int , int);

#endif