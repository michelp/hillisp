CUDA_ARCH_FLAGS := -arch=sm_30 -g -G
CC_FLAGS += $(CUDA_ARCH_FLAGS)

EXE = lisp

all: $(EXE)

% : %.cu
	nvcc device.cu lisp.cu $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean: 
	rm -f $(EXE)
