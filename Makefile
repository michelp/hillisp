CUDA_ARCH_FLAGS := -arch=sm_30 -g -G
CC_FLAGS += $(CUDA_ARCH_FLAGS)

EXE = lisp

all: $(EXE)

% : %.cu
	nvcc --relocatable-device-code=true ns.cu core.cu gc.cu flow.cu io.cu cmp.cu math.cu lisp.cu $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean: 
	rm -f $(EXE)
