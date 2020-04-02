## Mandelbrot renderer in CUDA

My lazy CUDA implementation of the escape time algorithm with renormalization for zooming into the mandelbrot set. Generates 1024x1024 .pgm files as frame output. The the script uses ffmpeg to convert them to a video. Since .pgm files stores the raw matrix as image, the generated images can take up to 30 gbs ! (Need to fix this later) 

![Mandelbrot Set Zooming](mandelbrot.gif)

## Usage

~~~~
mkdir frames
make
make run
./make_video.sh
~~~~

## Machine Details

Tested on a machine with the following configuration:
- Ubuntu 18.04
- Cuda Driver Version 10.2
- GeForce GTX 1660 Ti - 6 gb

## References
1. https://www.kth.se/social/files/5504b42ff276543e4aa5f5a1/An_introduction_to_the_Mandelbrot_Set.pdf
2. http://linas.org/art-gallery/escape/escape.html