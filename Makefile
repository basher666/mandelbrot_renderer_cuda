CC = nvcc
render: kernel.cu host.cu
	$(CC) -I ./ kernel.cu host.cu -o render_set
run: render_set
	./render_set
clean:
	rm render_set