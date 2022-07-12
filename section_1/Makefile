CC = nvcc
PROGRAMS = kernel hello_cuda organization_of_threads unique_idx_calc_threadIdx

all: $(PROGRAMS)

%: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(PROGRAMS)