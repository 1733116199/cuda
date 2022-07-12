CC = nvcc
PROGRAMS = kernel hello_cuda organization_of_threads

all: $(PROGRAMS)

%: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(PROGRAMS)