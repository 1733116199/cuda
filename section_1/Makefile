CC = nvcc
PROGRAMS = kernel \
hello_cuda organization_of_threads \
unique_idx_calc_threadIdx \
index_calculation_for_2d_grid \
index_calculation_for_2d_grid2 \
memory_transfer \
sum_array_gpu

all: $(PROGRAMS)

%: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(PROGRAMS)