CC = g++
NVCC = nvcc

render: kernel.o main.o
	$(CC) kernel.o main.o -L/usr/local/cuda/lib64 -lcudart -o render
main.o: main.cpp
	$(CC) -c -I. main.cpp -o main.o
kernel.o: kernel.cu kernel.h
	$(NVCC) -c -I. -I/usr/local/cuda/include kernel.cu -o kernel.o
run: render
	./render
clean:
	rm -f render kernel.o main.o
