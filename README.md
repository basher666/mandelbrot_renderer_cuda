## Mandelbrot renderer in CUDA

A simple CUDA implementation of the escape time algorithm for mandelbrot set. Generates a 1024x1024 .pgm file as output. Followed the excellent tutorial https://www.kth.se/social/files/5504b42ff276543e4aa5f5a1/An_introduction_to_the_Mandelbrot_Set.pdf for understanding Mandelbrot sets.

![Mandelbrot Set Coloured](mandelbrot.jpeg)

## Usage

> make
> make run

## Machine Details

Tested on a machine with the following configuration:
- Ubuntu 18.04
- Cuda Driver Version 10.2
- GeForce GTX 1660 Ti - 6 gb