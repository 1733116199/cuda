kernel: kernel.cu
	nvcc -o kernel kernel.cu
	./kernel
hello_cuda: hello_cuda.cu
	nvcc -o hello_cuda hello_cuda.cu
	./hello_cuda
clean:
	rm -rf kernel hello_cuda